"""
Configuration settings for the memory system.
"""
import os
from prompts import get_prompt

class Config:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        if self._config is not None:
            return

        # Default configuration values
        self._config = {
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO")  # Can be DEBUG, INFO, WARNING, ERROR, CRITICAL
            },
            "memory": {
                "short_term": {
                    "read": {
                        "fetch_limit": 5,                    # Number of short term memory turns to fetch
                        "reset_on_context_shift": False     # Reset short term memory when context changes
                    },
                    "write": {
                        "saved_turns": 10,                     # Number of short term memory turns to save
                    }
                },
                "long_term": {
                    "read": {
                        "similar_nodes_fetch_limit": 5,          # Number of similar nodes to fetch via embedding similarity
                        "retrieval_query_num_phrases": 5,          # Number of phrases to generate from prompt and context for relavant nodes similarity search
                        "node_search_similarity_threshold": 0.7,   # Sets the threshold for node search by embeddings similarity
                        "node_traverse_similarity_threshold": 0.8, # Sets the threshold for node traversal by embeddings similarity,
                        "node_include_top_percentile": 0.60,      # Sets the top percentile of nodes to include in the context (e.g.,  60% of top similarity score)
                        "node_include_min_node_number": 5,       #Include all nodes if the number of nodes is less than this number
                        "strong_similarity_score": 0.6,         # Defines the "strong" match quality of a retrieved node to be included in the context
                        "normal_similarity_score": 0.3,         # Defines the "normal" match quality of a retrieved node to be included in the context
                    },
                    "write": {
                        "enabled": True, # Enable long term memory storage
                        "synthesis_max_tokens": 500,          # Max tokens for the synthesis prompt
                        "synthesis_min_nodes": 3,              # Min number of nodes to synthesize into a new node
                        "synthesis_max_nodes": 10,              # Max number of nodes to synthesize into a new node
                        "new_node_max_concepts": 5,                 # Max number of concepts to generate for the new node (later to be concept nodes)
                        "new_concepts_max_tokens": 180,         # Max tokens for the new concepts list created for a new node.
                        "node_title_max_tokens": 60,         # Max tokens for the node title
                        "node_to_node_link_similarity_threshold": 0.8,  # Minimum similarity needed to find new memory or knowledge node to link to the new node
                        "concept_node_search_fetch_limit": 5,      # Number of similar nodes to fetch via embedding similarity
                        "concept_node_search_similarity_threshold": 0.9,  # Minimum similarity needed to find similar concept node to be reused instead of creating new concept node.
                        "concept_weight_in_embedding": 0.6,  # Weight of the concept phrase vs. the parent node text in the formed embedding
                        
                    },
                },   
                "optimize_message_list": {
                    "enabled": True,
                    "include_short_term": False,
                    "optimized_context_max_tokens": 1000
                }
            },
            "embeddings": {
                "model": "all-MiniLM-L6-v2",
                "dimensions": 384,
                "vector_index_name": "unified_embeddings",
            },
            "llm": {
                "functionality": {
                    "main_chat_loop": {
                        "provider": os.getenv("LLM_MAIN_PROVIDER", "openai"),
                        "model": os.getenv("LLM_MAIN_MODEL", "gpt-4o")
                    },
                    "auxiliary": {
                        #"provider": os.getenv("LLM_AUX_PROVIDER", "localai"),
                        #"model": os.getenv("LLM_AUX_MODEL", "mistral-7b-instruct-v0.3")
                        "provider": os.getenv("LLM_MAIN_PROVIDER", "openai"),
                        "model": os.getenv("LLM_MAIN_MODEL", "gpt-4o-mini")
                    }
                },
                "localai": {
                    "host": os.getenv("LLM_HOST", "localhost"),
                    "port": int(os.getenv("LLM_PORT", "8080")),
                    "context_window": 8192,
                    "response_reserve": 1000,
                    "assistant_response_max_tokens": 500
                },
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "org_id": os.getenv("OPENAI_ORG_ID", ""),
                    "context_window": 32000,
                    "response_reserve": 1000,
                    "assistant_response_max_tokens": 600
                }
            },
            "db": {
                "provider": os.getenv("DB_PROVIDER", "neo4j"),  # Currently only 'neo4j' is supported
                "name": os.getenv("DB_NAME", "neo4j"),
                "neo4j": {
                    "host": os.getenv("NEO4J_HOST", "neo4j"),
                    "port": os.getenv("NEO4J_PORT", "7687"),
                    "user": os.getenv("NEO4J_USER", "neo4j"),
                    "password": os.getenv("NEO4J_PASSWORD", "testpassword")
                }
            },
            "web_server": {
                "host": os.getenv("WEB_SERVER_HOST", "0.0.0.0"),
                "port": os.getenv("WEB_SERVER_PORT", "5000"),
                "upload_folder": os.getenv("WEB_SERVER_UPLOAD_FOLDER", "uploads")
            },
            "prompts": {
                "system_prompt": get_prompt("reasoning-system-prompt")  # The prompt text to use
            }
        }
        
        # Config loaded

    def get(self, *keys):
        """Get a configuration value using dot notation."""
        value = self._config
        for key in keys:
            value = value[key]
        return value