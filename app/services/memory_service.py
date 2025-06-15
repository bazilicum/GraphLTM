"""
Service for memory operations and queries.
"""
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from prompts import get_prompt        
from clients.llm_client import BaseLLMClient
from clients.db_client import BaseDBClient
from utils.embedding import generate_embedding, generate_mixed_embedding, count_tokens
import numpy as np
from utils.conversation_history import ConversationHistory
from utils.logging_utils import setup_logging
from config import Config

class ContextRetrievalService:
    def __init__(self, db_client: BaseDBClient, main_llm_client: BaseLLMClient, aux_llm_client: BaseLLMClient, config=None, logger=None):
        self.db_client = db_client
        self.main_llm_client = main_llm_client
        self.aux_llm_client = aux_llm_client
        #load config and setup logging
        self.config = config or Config()
        self.logger = logger or setup_logging(self.config, __name__)
        self.conversation_history = ConversationHistory.get_instance(
            max_turns=self.config.get('memory', 'short_term', 'write', 'saved_turns')
        )

    def determine_context_relation(self, prompt: str, memory_turns: List[Dict[str, str]]) -> int:
        """
        Determine the numeric context relationship score between the latest user prompt and the previous assistant response.
        
        Args:
            prompt: The user's current prompt
            memory_turns: List of previous conversation turns, each containing 'user' and 'assistant' messages
            
        Returns:
            An integer score between RETRIEVAL_QUERY_MIN_SCORE (not related) and RETRIEVAL_QUERY_MAX_SCORE (highly related)
        """
        last_turn = memory_turns[-1] if memory_turns else None
        score = 1  # Default score for unrelated if no previous turn

        if last_turn:
            decision_prompt = (
                "You are a language model specialized in identifying if the USER PROMPT is "
                "a standalone sentence or it relates to the previous ASSISTANT RESPONSE.\n\n"
                "Provide a numeric value between 1 to 10 where 1 is not related and 10 is related.\n\n"
                f"ASSISTANT RESPONSE: \"{last_turn['assistant']}\"\n"
                f"USER PROMPT: \"{prompt}\"\n\n"
                "Return *only* with the following JSON format:\n"
                "{\"score\": <1-10>}"
            )
            response = self.aux_llm_client.generate(decision_prompt).strip()
            response = response.replace("\n", "")
            match = re.search(r'\{.*?\}', response)
            if match:
                try:
                    parsed = json.loads(match.group())
                    score = int(parsed.get("score", 1))
                    self.logger.info(f"Prompt relation to the previous turn where 1 is not related and 10 is very related: {score}")
                except Exception as e:
                    self.logger.warning(f"Failed to parse context score: {e}")
        return score



    def generate_retrieval_query(self, prompt: str, memory_turns: List[Dict[str, str]], num_phrases: int) -> Optional[List[str]]:
        """
        Generate a list of Wh-questions for similarity search in long-term memory.
        
        Args:
            prompt: The user's current prompt
            memory_turns: List of previous conversation turns, each containing 'user' and 'assistant' messages
            num_phrases: Maximum number of questions to generate
            
        Returns:
            List of Wh-questions (who, what, where, when, why, how, which) or None if generation fails
        """
        memory_context = "\n".join(
            [f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in memory_turns[-3:]]
        )
        full_input = (
            "Your response token budget is 120.\n\n"
            f"You are a language model specialized in identifying and generating up to {str(num_phrases)} high-quality, contextually relevant, "
            "and informative **Wh-questions** (who, what, where, when, why, how, which) based on a user's message and related context.\n\n"
            "These questions will be used for similarity search over a long-term memory graph.\n"
            "Each question should reflect an *implicit or explicit need for information* in the user's prompt, "
            "and should be suitable for matching with memories or concepts that could serve as answers.\n\n"
            f"Recent conversation context:\n\"{memory_context}\"\n\n"
            f"New user message:\n\"{prompt}\"\n\n"
            "Return *only* the generated question as a list in a JSON format. Do not include any other output or explanation."
        )
        try:
            response = self.aux_llm_client.generate(full_input, max_tokens=120)
            self.logger.debug(f"Generated retrieval query response:\n{response}")

            # Robustly extract the first JSON array (even if pretty-printed)
            match = re.search(r'\[\s*(?:"(?:[^"\\]|\\.)*"\s*,?\s*)+\]', response, re.DOTALL)
            if not match:
                self.logger.warning(f"No JSON array found in response: {response}")
                return None

            questions = json.loads(match.group())
            if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
                self.logger.warning("Extracted JSON is not a list of strings")
                return None
            return questions

        except Exception as e:
            self.logger.error(f"Error extracting questions from LLM response: {e}")
            return None

    def retrieve_and_dedupe_points(self, queries: list) -> list:
        """
        For each query, fetch similar nodes from the database, deduplicate by node ID,
        and return a list of unique nodes sorted by similarity (highest first).
        """
        retrieved_points = []
        seen_nodes = set()
        fetch_limit = self.config.get('memory', 'long_term', 'read', 'similar_nodes_fetch_limit')
        min_similarity = self.config.get('memory', 'long_term', 'read', 'node_search_similarity_threshold')
        label_filter = ["Memory", "Synthesis", "Concept", "Knowledge"]

        for i, query in enumerate(queries):
            if query is None or not query.strip():
                self.logger.warning(f"Skipping empty query at index {i}")
                continue
            query_embedding = generate_embedding(query)
            points = self.db_client.fetch_similar_nodes(
                query_embedding=query_embedding,
                fetch_limit=fetch_limit,
                min_similarity=min_similarity,
                label_filter=label_filter
            )
            for point in points:
                node_id = point.get('id')
                if node_id and node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    retrieved_points.append(point)

        # Sort nodes by similarity score (highest first)
        retrieved_points.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return retrieved_points
    
    def evaluate_node_relevance(
        self,
        node: Dict[str, Any],
        similarity: float,
        min_similarity: float,
        max_similarity: float,
        number_of_retrieved_nodes = int,
    ) -> Dict[str, Any]:
        """
        Evaluate a node's relevance based on where its similarity falls within
        the batch distribution.
        The purpose is to determine which nodes should be included in the final
        retrieval set and which should be traversed to find more relevant nodes.

        *   Nodes in the top `top_pct` percentile (default 60%) are included.
        *   Nodes in the top 20% are also traversed.
        *   Embedding strength is mapped to quantiles.

        Args:
            node:           The node dictionary.
            similarity:     Node's raw similarity score.
            min_similarity: Minimum similarity score in the batch.
            max_similarity: Maximum similarity score in the batch.
            top_pct:        Fraction (0-1) of upper-quantile nodes to keep.
        """
        labels = node.get("labels", [])

        # Normalize similarity to a 0-1 percentile within this batch
        spread = max(max_similarity - min_similarity, 1e-5)
        percentile = (similarity - min_similarity) / spread      # 0 = worst, 1 = best
        top_pct = self.config.get('memory', 'long_term', 'read', 'node_include_top_percentile')
        min_node_number = self.config.get('memory', 'long_term', 'read', 'node_include_min_node_number')
        include   = number_of_retrieved_nodes<min_node_number or percentile >= (1.0 - top_pct)                # e.g., top 60%
        traverse  = percentile >= self.config.get('memory', 'long_term', 'read', 'node_traverse_similarity_threshold')      

        if percentile >= self.config.get('memory', 'long_term', 'read', 'strong_similarity_score'):
            embed_strength = "strong"
        elif percentile >= self.config.get('memory', 'long_term', 'read', 'normal_similarity_score'):
            embed_strength = "normal"
        else:
            embed_strength = "weak"
            include = False
        self.logger.debug(f"Node: {labels} embed_strength: {embed_strength}, include: {include}, traverse: {traverse}")


        return {
            "include": include,
            "traverse": traverse,
            "embed_strength": embed_strength,
            "node_id": node.get("id"),
            "labels": labels,
            "similarity": similarity,
            "percentile": round(percentile, 2)
        }

    def render_turn_from_node(self, node_label: str, node_payload: Dict[str, Any]) -> List[Dict[str, str]]:
        """Return one or two chat messages for a node."""
        if node_label == "Knowledge":
            source = node_payload.get("source_name", "unknown source")
            file_  = node_payload.get("source_file", "unknown file")
            page   = node_payload.get("page_num", "?")
            text   = node_payload.get("text", "")
            self.logger.debug(f"Retrieving text of knowledge node id:{node_payload.get('id')}")
            return [{
                "role": "assistant",
                "content": f"[knowledge from {source} (file: {file_}, page {page})] {text}"
            }]
        elif node_label == "Synthesis":
            self.logger.debug(f"Retrieving text of synthesis node id:{node_payload.get('id')}")
            return [{
                "role": "assistant",
                "content": f"[synthesis ({node_payload.get('title', '')})] {node_payload.get('synthesis_text', '')}"
            }]
        else:  # Memory
            self.logger.debug(f"Retrieving prompt and response of memory node id:{node_payload.get('id')}")
            return [
                {"role": "user",
                    "content": f"[memory] {node_payload.get('prompt', '')}"},
                {"role": "assistant",
                    "content": node_payload.get("response", "")}
            ]

    def compose_context_from_relevance_plan(
        self,
        nodes: List[Dict[str, Any]],
        recent_memories: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Compose context messages from a list of evaluated nodes, deduplicating recent memories and optionally traversing.

        Args:
            nodes: List of retrieved nodes (including similarity scores).
            recent_memories: List of recent memory turns, each containing 'user' and 'assistant' messages.
            db_client: Database client used for traversal.

        Returns:
            Updated message list with context messages.
        """
        # helper: reserve tokens and *return* the message dict or None
        def reserve_message(role: str, content: str) -> Dict[str, str] | None:
            """
            If the message fits under the remaining token budget, reserve the
            tokens and return the message dict; otherwise return None.
            """
            nonlocal current_tokens
            t = count_tokens(content)
            if current_tokens + t <= max_tokens:
                current_tokens += t
                return {"role": role, "content": content}
            return None

        # Token budget
        provider = self.config.get('llm', 'functionality', 'main_chat_loop', 'provider')
        max_tokens = self.config.get("llm", provider, "context_window") - self.config.get("llm", provider, "response_reserve")
        current_tokens = 0

        # Start the message list with the system role message
        message_list = []
        message_list.append(reserve_message("system",self.config.get("prompts","system_prompt")))
        self.logger.debug(f"Budget permits the system message. Added system role message to message list. {max_tokens-current_tokens} left.")

        # Convert recent_memories to the expected format for recent_ids
        recent_ids = set()
        stm_list = []
        # Add recent memories to temp list. This is done here to make sure they make it into the context window budget
        for mem in recent_memories:
            if isinstance(mem, dict):
                recent_ids.add(mem['id'])
                m_user = reserve_message("user", mem.get('user', ''))
                m_assistant = reserve_message("assistant", mem.get('assistant', ''))
                if m_user and m_assistant:
                    stm_list.extend([m_user, m_assistant])
                    self.logger.debug(f"Budget permits the recent memory: {mem['id']}. {max_tokens-current_tokens} left.")
                else:
                    break               

        max_similarity = max((n.get("similarity", 0) for n in nodes), default=1.0)
        min_similarity = self.config.get("memory", "long_term", "read", "node_search_similarity_threshold")

        all_nodes_to_include = {}
        
        for node in nodes:
            node_id = node.get("id")
            if node_id in recent_ids:
                continue

            # For every node retrieved from the similarity search, suggest with evaluate_node_relevance what to do with it
            plan = self.evaluate_node_relevance(
                node= node,
                similarity= node.get("similarity", 0),
                min_similarity= min_similarity,
                max_similarity= max_similarity,
                number_of_retrieved_nodes= len(nodes)
            )

            if not plan["include"]:
                continue

            all_nodes_to_include[node_id] = node
            self.logger.debug(f"Considering node {node.get('labels', [])} as candidate for message list")

            if plan["traverse"]:
                # If the node is traversable, fetch associated nodes
                related_ids = self.db_client.traverse_related_nodes([node_id],target_labels=["Memory","Synthesis","Knowledge"],max_depth=1)
                for related_id in related_ids:
                    if related_id in recent_ids or related_id in all_nodes_to_include:
                        continue
                    related_node = self.db_client.get_node_by_id(related_id)
                    if related_node:
                        all_nodes_to_include[related_id] = related_node
                self.logger.debug(f"Considering additional {len(related_ids)} related nodes from traversing {node.get('labels', [])} as candidates for message list")

        # Organize nodes by type
        grouped = {"Knowledge": [], "Synthesis": [], "Memory": []}

        for node in all_nodes_to_include.values():
            labels = node.get("labels", [])
            for label in grouped:
                if label in labels:
                    grouped[label].append(node)
                    break

        # Sort each group by descending similarity
        for label in grouped:
            grouped[label].sort(key=lambda n: n.get("similarity", 0.0), reverse=True)


        # Select nodes (high-similarity first) but place them so that the
        # strongest evidence ends up closest to the user prompt.
        for label in ("Synthesis","Knowledge","Memory"):
            accepted: List[Dict[str, str]] = []     # stack for this bucket

            for node in grouped[label]:             # already sorted high to low similarity
                payload = node.get("node", {})
                msgs = self.render_turn_from_node(label, payload)
                role_one = reserve_message(msgs[0]["role"], msgs[0]["content"])
                if len(msgs)>1:
                    role_two = reserve_message(msgs[1]["role"], msgs[1]["content"])
                    if role_one and role_two:
                        accepted.extend([role_two,role_one])
                        self.logger.debug(f"Budget permits the {label} node id: {node.get('id')}. {max_tokens-current_tokens} left.")
                else:
                    if role_one is not None:
                        accepted.append(role_one)
                        self.logger.debug(f"Budget permits the {label} node id: {node.get('id')}. {max_tokens-current_tokens} left.")
                    else:
                        self.logger.debug(f"Context window is full: skipping lower priority nodes")
                        break

            # place most-similar messages *closest* to the user prompt
            message_list.extend(reversed(accepted))
            self.logger.debug(f"Adding {len(accepted)} role messages from {label} nodes from LTM to message list")

        # Add recent memories
        #message_list.extend(stm_list)
        #logger.debug(f"Adding {len(stm_list)} role messages from STM nodes to message list")

        return message_list, stm_list

    def reshape_context(self, prompt: str, context: str, max_tokens: int) -> str:
        """
        Reshape the retrieved context to be more concise and relevant.
        
        Args:
            llm_client: The LLM client to use
            context: Context to reshape
            
        Returns:
            Reshaped context
        """
        try:
            # Use the system prompt for context optimization
            reshape_prompt = (
                get_prompt("long-term-memory-context-optimization") +
                "\n\n" 
                "[LONG_TERM_MEMORY]\n" 
                f"{context}\n\n" 
                "[USER_PROMPT]\n" 
                f"{prompt}\n\n"
            )
            
            # Generate optimized context
            reshaped_context = self.main_llm_client.generate(reshape_prompt, max_tokens)
            self.logger.debug(f"Reshaped retrieved context successfully")
            return reshaped_context
        except Exception as e:
            self.logger.error(f"Error reshaping context: {e}; returning original context")
            return context

    def optimize_message_list(self, message_list: List[Dict[str, str]], user_prompt: str) -> List[Dict[str, str]]:
        """
        Optimize the message list by compressing and restructuring based on the user prompt.
        This function is gated by configuration and only runs if enabled.
        
        Args:
            message_list: List of messages to optimize
            user_prompt: Current user prompt
            config: Configuration instance
            
        Returns:
            Optimized message list if enabled, otherwise returns the original list
        """
        # Analyze message relevance to current prompt
        compressed_messages = []
        context_for_compression = []
        
        for message in message_list:
            if message.get('role') == 'system':
                # Always keep system messages
                compressed_messages.append(message)
            else:
                context_for_compression.append(message)

        # Combine remaining messages into a single bucket
        if context_for_compression:
            # Combine messages in chronological order
            combined_content = " ".join(msg['content']+"\n" for msg in context_for_compression)

            # Add the combined message
            compressed_messages.append({
                'role': 'assistant',  # Use the role of the most recent message
                'content': "[PRELIMINARY CONTEXT BASED ON LONG-TERM MEMORY]\n"+self.reshape_context(user_prompt, combined_content, self.config.get('memory', 'optimize_message_list', 'optimized_context_max_tokens'))
            })

        # Reverse to maintain original chronological order
        return compressed_messages

    def process_context_retrieval (self, user_prompt: str):
        # Get recent conversation history from in-memory STM
        self.logger.info(f"Getting recent conversation history from in-memory STM")
        recent_turns = self.conversation_history.get_recent_turns(self.config.get('memory', 'short_term', 'read', 'fetch_limit'))
        self.logger.info(f"Recent turns count: {len(recent_turns)}")
        
        # Determine context relation and rephrase prompt if needed
        if self.config.get('memory', 'short_term', 'read', 'reset_on_context_shift'):
            self.logger.info(f"Determining prompt relation to the previous turns")
            context_relation = "new" if self.determine_context_relation (user_prompt, recent_turns)<5 else "related"
            self.logger.info(f"Prompt relation to the previous turns: {context_relation}")
            # If the prompt is not related to the last turns, clear the history to create accurate retrival terms and dynamic context window
            recent_turns = recent_turns if context_relation == "related" else []


        # Generate retrieval query and embeddings for each question
        self.logger.info(f"Generating retrieval query")
        retrieval_queries = self.generate_retrieval_query(
            prompt=user_prompt, 
            memory_turns=recent_turns, 
            num_phrases=self.config.get('memory', 'long_term', 'read', 'retrieval_query_num_phrases'))

        if not isinstance(retrieval_queries, list):
            self.logger.warning("Retrieval query did not return a list of queries, using the original prompt")
            retrieval_queries = [user_prompt]

        # Query database for each question and merge results
        self.logger.info(f"Querying database for {len(retrieval_queries)} questions")
        retrieved_points = self.retrieve_and_dedupe_points(queries=retrieval_queries)
        self.logger.info(f"Combined retrieved points: {len(retrieved_points)} unique nodes")
        
        # Sort nodes by similarity score (highest first)
        retrieved_points.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        
        # Compose context from relevance plan
        self.logger.info(f"Generating context from retrieved nodes")
        ltm_message_list,stm_message_list = self.compose_context_from_relevance_plan(
            nodes=retrieved_points,
            recent_memories=recent_turns
        )
        # Selects which part of the message list should be optimized
        if self.config.get('memory', 'optimize_message_list', 'enabled'):
            if self.config.get('memory', 'optimize_message_list', 'include_short_term'): 
                # Optimize message list with both long-term and short-term messages
                message_list=ltm_message_list+stm_message_list
                self.logger.info(f"Optimizing message list")
                message_list = self.optimize_message_list(message_list, user_prompt)
            else:
                # Optimize only long-term messages
                self.logger.info(f"Optimizing message list")
                message_list = self.optimize_message_list(ltm_message_list, user_prompt)
                message_list.extend(stm_message_list)
        else:
            message_list=ltm_message_list+stm_message_list

        return message_list


class ContextStoringService:
    def __init__(self, db_client: BaseDBClient, main_llm_client: BaseLLMClient, aux_llm_client: BaseLLMClient, config=None, logger=None):
        self.db_client = db_client
        self.main_llm_client = main_llm_client
        self.aux_llm_client = aux_llm_client
        #load config and setup logging
        self.config = config or Config()
        self.logger = logger or setup_logging(self.config, __name__)
        self.conversation_history = ConversationHistory.get_instance(
            max_turns=self.config.get('memory', 'short_term', 'write', 'saved_turns')
        )

    def should_store_memory(self, conversation_turn: str) -> bool:
        """
        Determines if a conversation turn should be saved in long-term memory.
        
        Args:
            ai_client: LLM client
            conversation_turn: Conversation turn to evaluate
            max_tokens: Maximum tokens for the response
            
        Returns:
            True if the memory should be stored, False otherwise
        """
        gating_prompt = (
            "Please carefully analyze the following conversation turn and provide two ratings on a scale of 1 to 10:\n"
            "1. 'info': Rate the factual and informational value of this conversation turn. Consider how clear, detailed, "
            "and relevant the content is, where 1 means trivial/insubstantial and 10 means highly informative with "
            "significant new details.\n"
            "2. 'emotion': Rate the emotional impact or intensity expressed in this conversation turn. Consider both the "
            "emotional tone and the depth of expressed feeling, where 1 means no emotional content and 10 means very "
            "high emotional intensity.\n\n"
            "Here is the conversation turn:\n"
            f"{conversation_turn}\n\n"
            "Make sure you assess the content critically without defaulting to standard values. "
            "Provide your answer strictly in JSON format with exactly two keys, like this:\n"
            "{\"info\":<1-10>, \"emotion\":<1-10>}"
        )
        
        try:
            response = self.aux_llm_client.generate(gating_prompt, max_tokens=20)
            response = response.replace('\n', '')
            
            # Use a regular expression to capture the JSON object in the response
            match = re.search(r'\{.*\}', response)
            if not match:
                self.logger.warning("Could not find JSON in memory store evaluation response")
                return False

            data = json.loads(match.group())
            
            # Check if either the informational value or emotional value meets/exceeds 7
            info_value = data.get("info", 0)
            emotion_value = data.get("emotion", 0)
            
            should_store = info_value >= 7 or emotion_value >= 7
            self.logger.info(f"Memory store evaluation: info={info_value}, emotion={emotion_value}, store={should_store}")
            return should_store
        
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON from memory store evaluation: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in memory store evaluation: {e}")
            return False

    def generate_title(self, text: str) -> str:
        """
        Generate a title for a text using the LLM.
        
        Args:
            ai_client: The LLM client to use
            text: Text to generate a title for
            max_tokens: Maximum tokens for the title
            
        Returns:
            Generated title
        """
        title_prompt = (
            "You are a language model specialized in identifying and extracting one high-quality, contextually relevant, "
            "and informative search phrases from a given text.\n\n"
            f"{text}\n\n"
            "Your goal is to help the user find accurate and detailed information about the topics discussed in the "
            "recent conversation by generating concise and specific search phrase that can be used in a search engine."
            "return *only* the search phrase, nothing more."
        )
        
        try:
            max_tokens = self.config.get("memory", "long_term", "write", "node_title_max_tokens")
            title = self.aux_llm_client.generate(title_prompt, max_tokens)
            self.logger.debug(f"Generated title: {title}")
            return title
        except Exception as e:
            self.logger.error(f"Error generating title: {e}")
            return "Untitled"

    def create_synthesis(self, unsynthesized_memories: List[Dict[str, str]]) -> str:
        """
        Create a synthesis from unsynthesized memories.
        
        Args:
            unsynthesized_memories: List of unsynthesized memories
            db_client: Database client
            llm_client: LLM client
            
        Returns:
            ID of the created synthesis node
        """
        
        try:
            # Combine all texts from memories
            all_text = " ".join([item["text"] for item in unsynthesized_memories])
            
            # Generate synthesis text
            synthesis_prompt = f"You are a researcher. Create an article from the following conversation content:\n\n{all_text}"
            synthesis_text = self.aux_llm_client.generate(synthesis_prompt, self.config.get("memory", "long_term", "write", "synthesis_max_tokens"))
            
            # Generate title and embedding
            synthesis_title = self.generate_title(synthesis_text)
            synthesis_embedding = generate_embedding(synthesis_text)
            
            # Create synthesis node
            synthesis_id = self.db_client.create_synthesis_node(
                synthesis_text, 
                synthesis_embedding, 
                synthesis_title
            )
            self.logger.info(f"Created synthesis node: {synthesis_title} id:{synthesis_id}")
            
            return synthesis_id
        except Exception as e:
            self.logger.error(f"Error creating synthesis: {e}")
            return None

    def check_synthesis_trigger(self, connected_node_ids, unsynthesized_nodes, memory_id: str) -> bool:
        """
        Organically synthesize memory nodes when a new node deviates from the current cluster.
        """

        if len(unsynthesized_nodes) < self.config.get("memory", "long_term", "write", "synthesis_min_nodes"):
            return False  # Not enough nodes for synthesis in the last turn cluster
        elif len(unsynthesized_nodes) > self.config.get("memory", "long_term", "write", "synthesis_max_nodes"):
            return True  # Too many nodes, in the last turn cluster trigger synthesis

        if not memory_id in connected_node_ids:
            return True  # New memory node is not related to the last node cluster thus diviates. Last turn cluster needs to synthesize

        return False

    def process_synthesis_creation(self, memory_id: str, last_turn: Dict[str, Any]) -> str | None:
        """
        Process synthesis for a new memory node if needed.
        
        Args:
            memory_id: ID of the new memory node
            db_client: Database client
            llm_client: LLM client
        """
        
        if not last_turn or len(last_turn)==0:
            self.logger.debug("No recent conversation turns found. Cannot process synthesis for the previous turn.")
            return None
        
        last_turn = last_turn[0]  # Get the last turn from the list
        
        # Traverse graph to get all connected nodes of the last turn via RELATES_TO edges. 
        connected_node_ids = self.db_client.traverse_related_nodes([last_turn["id"]],start_labels=["Memory"],target_labels=["Memory"],edge_types=["RELATES_TO"])

        # Filter only unsynthesized ones
        unsynthesized = self.db_client.get_unsynthesized_related_nodes(connected_node_ids)

        # check if we have enough context drift in memories to initiate a synthesis
        if self.check_synthesis_trigger(connected_node_ids, unsynthesized, memory_id):
            # Create synthesis for the previous unsynthesized memories
            synthesis_id = self.create_synthesis(unsynthesized)
            
            if synthesis_id:
                # Link synthesis to unsynthesized memories
                self.db_client.link_unsynthesized_memory_to_synthesis_nodes(synthesis_id,unsynthesized)
                self.logger.debug(f"Linked cohesive unsynthesized memories of the last turn to synthesis. synthesis_id:{synthesis_id}")
                # Link synthesis to other synthesis nodes in the cluster
                self.db_client.link_extending_synthesis_nodes(synthesis_id,last_turn["id"])
                self.logger.debug(f"Linked synthesis to other synthesis nodes.")
                return synthesis_id
        else:
            self.logger.debug(f"No cohestion drift detected nor max limit met in the correcnt ({len(unsynthesized)}) unsynthesized nodes. No synthesis created.")
        
        return None

    def extract_key_concepts(self, prompt: str,  num_keys: int = 5, max_tokens: int = 180) -> List[str]:
        """
        Extract key concepts or phrases from the user prompt using chat-based LLM API.
        
        Args:
            llm_client: The LLM client to use
            prompt: The text from which to extract concepts
            num_keys: Number of key concepts to extract
            max_tokens: Max tokens to return from the model
        
        Returns:
            A list of key concept phrases
        """
        system_instruction = (
            f"Extract a list of {num_keys} high-value key terms from the given text."
            "These terms should be multi-word phrases (2-8 words) that are either"
            "specific entities, technical concepts, or domain terminology, or phrases"
            "that represent the main idea or focus of the text,"
            ", or phrases that are frequently used in in the text."
            "Additionally, section headings, chapter headings, subheadings, and any other important headings"
            "or titles should be included as key terms."
            "Prioritize terms that are most likely to be used in a search. Do not exceed the list limits."
            "Return them as a JSON array of strings, with no extra text."
        )
        messages = [
            {"role": "system", "content": system_instruction}
        ]

        try:
            response_data = self.aux_llm_client.chat_generate(
                prompt=prompt,  # No extra prompt, just use the messages
                message_list=messages,
                max_tokens=max_tokens
            )
            response = response_data.get("content", "").replace('\n', '')

            # Extract JSON array using regex
            match = re.search(r'\[.*?\]', response)
            if not match:
                self.logger.warning("Failed to extract key concepts from model response.")
                return []
            self.logger.debug(f"Extracted key concepts:\n{match.group()}")
            return json.loads(match.group())

        except Exception as e:
            self.logger.error(f"Error extracting key concepts: {e}")
            return []

    def process_concept_from_new_node(self, text: str, node_id: str, parent_node_embedding: list):
            """
                Extract and process concept nodes linked to a node.
                Args:
                    text: The text to extract concepts from
                    node_id: ID of the target node
                    parent_node_embedding: Embedding of the parent node
            """
            # Extract key concepts from the text and process concept nodes
            concepts = self.extract_key_concepts(
                prompt=text,
                num_keys=self.config.get("memory", "long_term", "write", "new_node_max_concepts"),
                max_tokens=self.config.get("memory", "long_term", "write", "new_concepts_max_tokens"),
            )
            for phrase in concepts:
                concept_embedding = generate_mixed_embedding(generate_embedding(phrase), parent_node_embedding,self.config.get("memory", "long_term", "write", "concept_weight_in_embedding"))
                self.logger.debug(f"Generating embedding for concept: {phrase}")
                similar_concepts = self.db_client.fetch_similar_nodes(
                    query_embedding=concept_embedding,
                    fetch_limit=self.config.get("memory", "long_term", "write", "concept_node_search_fetch_limit"),
                    min_similarity=self.config.get("memory", "long_term", "write", "concept_node_search_similarity_threshold"),
                    label_filter=["Concept"]
                )
                self.logger.debug(f"Found {len(similar_concepts)} similar concepts")
                matched = False
                for node in similar_concepts:
                    if "Concept" in node["labels"]:
                        self.db_client.link_concept_to_node(node["id"], node_id)
                        matched = True
                        self.logger.debug(f"Linked existing concept: {node['node'].get('phrase')} to the newly created node: {node_id}")
                        break
                if not matched:
                    concept_id = self.db_client.create_concept_node(phrase, concept_embedding)
                    self.db_client.link_concept_to_node(concept_id, node_id)
                    self.logger.debug(f"Created new concept: {phrase} and linked to the newly created node: {node_id}")

    def process_context_storing(
        self,
        prompt: str,
        assistant_response: str) -> str:
        """
        Process and store a new memory node if it meets criteria.
        
        Args:
            prompt: User prompt
            assistant_response: Assistant response
            last_turn: Last turn in the conversation history
            
        Returns:
            ID of the new memory node if created, None otherwise
        """
        last_turn = self.conversation_history.get_recent_turns(1)
        conversation_chunk = f"{prompt}\n{assistant_response}"
        node_embedding = generate_embedding(conversation_chunk)
        points = self.db_client.fetch_similar_nodes(
                query_embedding=node_embedding,
                fetch_limit=1,
                min_similarity=0.98,
                label_filter=["Memory"]
            )
        if len(points) > 0:
            self.logger.info(f"Found similar memory node: {points[0]['node'].get('id')}. Not storing this turn in long-term memory.")
            return None
        # Check if this memory should be stored
        if self.should_store_memory(conversation_chunk):
            # Create a memory title
            memory_title = self.generate_title(assistant_response)

            # Create the new prompt embeddings
            conversation_chunk_embedding = generate_embedding(conversation_chunk)
            
            # Store in DB and create related edges
            new_node_id = self.db_client.create_memory_node(
                prompt=prompt,
                response=assistant_response,
                embedding=conversation_chunk_embedding,
                title=memory_title
            )
            self.logger.info(f"Created new memory node: {new_node_id}")
            
            self.process_concept_from_new_node(conversation_chunk, new_node_id, conversation_chunk_embedding)
            
            # Fetch top similar nodes and create edges
            similar_nodes = self.db_client.fetch_similar_nodes(
                query_embedding=conversation_chunk_embedding, 
                fetch_limit=50, 
                min_similarity=self.config.get('memory', 'long_term', 'write', 'node_to_node_link_similarity_threshold'),
                label_filter=["Memory","Knowledge"]
            )
            self.db_client.link_related_memory_nodes(new_node_id, similar_nodes)
            self.logger.info(f"Linked new memory nodes to similar nodes")
            
            # Process synthesis if needed
            self.process_synthesis_creation(new_node_id, last_turn)
            
            return new_node_id
        else:
            self.logger.info("Turn was not stored in long-term memory (did not meet criteria)")
            return None


class MemoryService:
    def __init__(self, db_client: BaseDBClient, main_llm_client: BaseLLMClient, aux_llm_client: BaseLLMClient, config=None, logger=None):
        self.db_client = db_client
        self.main_llm_client = main_llm_client
        self.aux_llm_client = aux_llm_client
        #load config and setup logging
        self.config = config or Config()
        self.logger = logger or setup_logging(self.config, __name__)
        self.context_retrieval_service = ContextRetrievalService(
            db_client=self.db_client,
            main_llm_client=self.main_llm_client,
            aux_llm_client=self.aux_llm_client,
            config=self.config,
            logger=self.logger
        )
        self.context_storing_service = ContextStoringService(
            db_client=self.db_client,
            main_llm_client=self.main_llm_client,
            aux_llm_client=self.aux_llm_client,
            config=self.config,
            logger=self.logger
        )
        
    def process_context_retrieval(self, user_prompt: str) -> List[Dict[str, str]]: 
        """
        Process context retrieval for a user prompt.
        Args:
            user_prompt: User prompt to process
        Returns:
            List of messages to be used as context for the LLM
        """
        return self.context_retrieval_service.process_context_retrieval(user_prompt)
    
    def process_context_storing(self, prompt: str, assistant_response: str) -> str | None:
        """
        Process context storing for a user prompt and assistant response.
        Args:
            prompt: User prompt
            assistant_response: Assistant response
        Returns:
            ID of the new memory node if created, None otherwise
        """
        return self.context_storing_service.process_context_storing(
            prompt=prompt,
            assistant_response=assistant_response
        )

    




