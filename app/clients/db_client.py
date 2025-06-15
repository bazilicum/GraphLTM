"""
Database clients and factory functions.
"""
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from abc import ABC, abstractmethod
from config import Config
from utils.logging_utils import setup_logging
import uuid

# Type aliases
NodeId = str
NodeLabel = str
EdgeType = str
Embedding = List[float]

# Global config and logger (consider moving to class level)
config = Config()
logger = setup_logging(config, __name__)

class BaseDBClient(ABC):

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def create_memory_node(self, prompt: str, response: str, embedding: Embedding, title: str = "") -> NodeId:
        """Create a memory node in the graph database."""
        pass

    @abstractmethod
    def create_synthesis_node(self, synthesis_text: str, embedding: Embedding, title: str = "") -> NodeId:
        """Create a synthesis node in the graph database."""
        pass

    @abstractmethod
    def create_knowledge_node(
        self,
        source_name: str,
        embedding: Embedding,
        source_file: str,
        page_num: int,
        chunk_id: int,
        text: str
    ) -> NodeId:
        """Create a knowledge node in the graph database."""
        pass

    @abstractmethod
    def create_concept_node(self, phrase: str, embedding: Embedding) -> NodeId:
        """Create a concept node in the graph database."""
        pass

    @abstractmethod
    def fetch_short_term_memory_nodes(self, duration_minutes: int = 10, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent memory nodes within the specified duration."""
        pass

    @abstractmethod
    def fetch_similar_nodes(self, query_embedding: Embedding, fetch_limit: int = 10, min_similarity: float = 0.7, label_filter: Optional[List[NodeLabel]] = None) -> List[Dict[str, Any]]:
        """Fetch nodes similar to the query embedding."""
        pass

    @abstractmethod
    def ensure_vector_index_exists(self, dimensions: int = 384) -> None:
        """Ensure the vector index exists in the database."""
        pass

    @abstractmethod
    def create_edge(self, edge_type: EdgeType, id1: NodeId, id2: NodeId, label1: NodeLabel = "Memory", label2: NodeLabel = "Memory", similarity: Optional[float] = None, source: str = "auto", confidence: Optional[float] = None) -> None:
        """Create an edge between two nodes."""
        pass

    @abstractmethod
    def link_related_memory_nodes(self, node_id: NodeId, similar_nodes: List[Dict[str, Any]]) -> None:
        """Link a node with its related memory nodes."""
        pass

    @abstractmethod
    def link_unsynthesized_memory_to_synthesis_nodes(self, synthesis_id: NodeId, unsynthesized_nodes: List[Dict[str, Any]]) -> None:
        """Link unsynthesized memory nodes to a synthesis node."""
        pass

    @abstractmethod
    def link_extending_synthesis_nodes(self, synthesis_id: NodeId, new_memory_id: NodeId) -> None:
        """Link two synthesis nodes."""
        pass

    @abstractmethod
    def get_unsynthesized_related_nodes(self, node_ids: List[NodeId]) -> List[Dict[str, Any]]:
        """Get related nodes that haven't been synthesized."""
        pass

    @abstractmethod
    def traverse_related_nodes(
        self,
        node_ids: List[NodeId],
        start_labels: Optional[List[NodeLabel]] = None,
        target_labels: Optional[List[NodeLabel]] = None,
        edge_types: Optional[List[EdgeType]] = None,
        max_depth: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Traverse related nodes in the graph."""
        pass

    @abstractmethod
    def link_concept_to_node(self, concept_id: NodeId, node_id: NodeId) -> None:
        """Link a concept node to another node."""
        pass

    @abstractmethod
    def fetch_edges_between_nodes(self, node_ids: List[NodeId]) -> List[Dict[str, Any]]:
        """Fetch edges between specified nodes."""
        pass

    @abstractmethod
    def get_node_by_id(self, node_id: NodeId) -> Dict[str, Any]:
        """Get node details by its ID."""
        pass

    @abstractmethod
    def get_sources(self) -> List[str]:
        """Get list of available sources."""
        pass

    @abstractmethod
    def delete_source(self, source_name: str) -> bool:
        pass

class Neo4jManager(BaseDBClient):
    """
    Manager for Neo4j database operations.
    """
    def __init__(self, host: str, port: str, user: str, password: str, db_name: str = None, vector_index_dimentions: int = 384):
        """
        Initialize the Neo4j manager.
        
        Args:
            uri: URI of the Neo4j database
            user: Username for authentication
            password: Password for authentication
            db_name: Name of the database (optional)
        """
        self.driver = GraphDatabase.driver(f"bolt://{host}:{port}", auth=(user, password))
        self.db_name = db_name
        logger.info(f"Initialized Neo4j connection to 'bolt://{host}:{port}' (db: {db_name})")

        # Ensure the vector index exists before any embedding operations
        try:
            self.ensure_vector_index_exists(dimensions=vector_index_dimentions)
        except Exception as e:
            logger.error(f"Failed to initialize vector index: {e}")
            raise

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        logger.info("Closed Neo4j connection")

    def create_memory_node(self, prompt: str, response: str, embedding: List[float], title: str = "") -> str:
        """
        Create a new memory node in Neo4j.
        
        Args:
            prompt: User prompt text
            response: Assistant response text
            embedding: Vector embedding of the memory
            title: Optional title for the memory
            
        Returns:
            ID of the created node
        """
        node_id = str(uuid.uuid4())
        
        try:
            with self.driver.session(database=self.db_name) as session:
                session.run(
                    """
                    CREATE (m:Memory:VectorSearchable {
                        id: $id,
                        prompt: $prompt,
                        response: $response,
                        embedding: $embedding,
                        title: $title,
                        created_at: datetime(),
                        last_used: datetime()
                    })
                    """,
                    id=node_id,
                    prompt=prompt,
                    response=response,
                    embedding=embedding,
                    title=title
                )
            logger.debug(f"Created memory node with ID: {node_id}")
            return node_id
        except Exception as e:
            logger.error(f"Error creating memory node: {e}")
            raise
    
    def create_synthesis_node(self, synthesis_text: str, embedding: List[float], title: str = "") -> str:
        """
        Create a new synthesis node in Neo4j.
        
        Args:
            synthesis_text: Text of the synthesis
            embedding: Vector embedding of the synthesis
            title: Optional title for the synthesis
            
        Returns:
            ID of the created node
        """
        node_id = str(uuid.uuid4())
        
        try:
            with self.driver.session(database=self.db_name) as session:
                session.run(
                    """
                    CREATE (s:Synthesis:VectorSearchable {
                        id: $id,
                        synthesis_text: $synthesis_text,
                        embedding: $embedding,
                        title: $title,
                        created_at: datetime(),
                        last_updated: datetime()
                    })
                    """,
                    id=node_id,
                    synthesis_text=synthesis_text,
                    title=title,
                    embedding=embedding
                )
            logger.debug(f"Created synthesis node with ID: {node_id}")
            return node_id
        except Exception as e:
            logger.error(f"Error creating synthesis node: {e}")
            raise
    
    def create_knowledge_node(
        self,
        source_name: str,
        embedding: List[float],
        source_file: str,
        page_num: int,
        chunk_id: int,
        text: str
    ) -> str:
        """
        Create a knowledge node in Neo4j from a chunk of a PDF.

        Args:
            source_name: Name of the source (human-readable or internal name)
            embedding: Vector embedding of the chunk
            source_file: Original filename of the PDF
            page_num: Page number in the PDF (1-indexed)
            chunk_id: ID of the chunk within the page
            text: Actual extracted chunk text

        Returns:
            ID of the created node
        """
        node_id = str(uuid.uuid4())

        try:
            with self.driver.session(database=self.db_name) as session:
                session.run(
                    """
                    CREATE (k:Knowledge:VectorSearchable {
                        id: $id,
                        source_name: $source_name,
                        source_file: $source_file,
                        page_num: $page_num,
                        chunk_id: $chunk_id,
                        text: $text,
                        embedding: $embedding,
                        created_at: datetime(),
                        last_updated: datetime()
                    })
                    """,
                    id=node_id,
                    source_name=source_name,
                    source_file=source_file,
                    page_num=page_num,
                    chunk_id=chunk_id,
                    text=text,
                    embedding=embedding
                )
            logger.debug(f"Created knowledge node with ID: {node_id}")
            return node_id
        except Exception as e:
            logger.error(f"Error creating knowledge node: {e}")
            raise

    def create_concept_node(self, phrase: str, embedding: List[float]) -> str:
        """
        Create a new concept node in Neo4j.
        
        Args:
            phrase: Concept phrase
            embedding: Vector embedding of the concept
            
        Returns:
            ID of the created node
        """
        node_id = str(uuid.uuid4())
        try:
            with self.driver.session(database=self.db_name) as session:
                session.run(
                    """
                    CREATE (c:Concept:VectorSearchable {
                        id: $id,
                        phrase: $phrase,
                        embedding: $embedding,
                        created_at: datetime()
                    })
                    """,
                    id=node_id,
                    phrase=phrase,
                    embedding=embedding
                )
            logger.debug(f"Created concept node with ID: {node_id}")
            return node_id
        except Exception as e:
            logger.error(f"Error creating concept node: {e}")
            raise

    # The function is deprected to support the disablment of LTM write 
    def fetch_short_term_memory_nodes(self, duration_minutes: int = 10, limit: int = 5) -> List[dict]:
        """
        Fetch recent memory nodes from Neo4j.
        
        Args:
            duration_minutes: Time window in minutes to consider
            limit: Maximum number of nodes to return
            
        Returns:
            List of memory node dictionaries
        """

        try:
            with self.driver.session(database=self.db_name) as session:
                # First, fetch nodes in the last N minutes
                recent_result = session.run(
                    """
                    CALL {
                    MATCH (m:Memory)
                    WHERE m.created_at >= datetime() - duration({minutes: $duration_minutes})
                    RETURN m
                    ORDER BY m.created_at DESC
                    LIMIT $limit
                    }
                    RETURN m
                    ORDER BY m.created_at ASC
                    """,
                    duration_minutes=duration_minutes,
                    limit=limit
                )
                nodes = [record["m"] for record in recent_result]

                # If none found, fetch the single last-created node
                if not nodes:
                    fallback_result = session.run(
                        """
                        MATCH (m:Memory)
                        RETURN m
                        ORDER BY m.created_at DESC
                        LIMIT 1
                        """
                    )
                    nodes = [record["m"] for record in fallback_result]

                logger.debug(f"Fetched {len(nodes)} short-term memory nodes")
                return nodes
        except Exception as e:
            logger.error(f"Error fetching short-term memory nodes: {e}")
            return []
        
    def fetch_similar_nodes(self, query_embedding: List[float], fetch_limit: int = 10, min_similarity: float = None, label_filter: List[str] = None) -> List[dict]:
        """
        Fetch nodes similar to the query embedding.
        
        Args:
            query_embedding: Vector embedding to search with
            fetch_limit: Maximum number of nodes to return
            min_similarity: Minimum similarity threshold
            label_filter: Optional list of labels to filter by (returns only nodes with these labels)
            
        Returns:
            List of node dictionaries with similarity scores
        """
        try:
            with self.driver.session(database=self.db_name) as session:
                # First, let's try with db.index.vector.queryNodes but with proper label filtering
                cypher = """
                CALL db.index.vector.queryNodes('unified_embeddings', $fetch_limit, $embedding)
                YIELD node, score
                """
                
                # Add label filtering if specified – runtime check so no warning if label absent
                if label_filter and len(label_filter) > 0:
                    cypher += "\nWHERE any(l IN labels(node) WHERE l IN $label_filter)"
                
                # Add minimum similarity filter if specified
                if min_similarity is not None:
                    if "WHERE" in cypher:
                        cypher += f"\nAND score >= $min_similarity"
                    else:
                        cypher += f"\nWHERE score >= $min_similarity"
                
                # Complete the query
                cypher += """
                RETURN node, labels(node) AS labels, score AS similarity, node.id AS id
                ORDER BY similarity ASC
                LIMIT $fetch_limit
                """
                
                similarity_result = session.run(
                    cypher, 
                    embedding=query_embedding,
                    fetch_limit=fetch_limit,
                    min_similarity=min_similarity,
                    label_filter=label_filter
                )
                
                # Process results within the session
                results = [
                    {
                        "source": "neo4j",
                        "similarity": record["similarity"],
                        "node": record["node"],
                        "labels": record["labels"],
                        "id": record["id"]
                    } for record in similarity_result
                ]
                
                logger.debug(f"Fetched {len(results)} similar nodes with label filter: {label_filter if label_filter else 'None'}")
                return results
        except Exception as e:
            logger.error(f"Error in vector similarity search: {e}")
            return []

    def ensure_vector_index_exists(self, dimensions: int = 384):
        """
        Ensure the vector index exists in Neo4j.
        
        Args:
            dimensions: Dimensionality of the embedding vectors
        """
        try:
            with self.driver.session(database=self.db_name) as session:
                # Check if the index already exists
                check_query = """
                SHOW INDEXES
                WHERE name = 'unified_embeddings' AND type = 'VECTOR'
                """
                result = session.run(check_query)
                
                # If index doesn't exist, create it
                if not result.peek():
                    create_query = """
                    CALL db.index.vector.createNodeIndex(
                        'unified_embeddings',
                        'VectorSearchable',
                        'embedding',
                        $dimensions,
                        'cosine'
                    )
                    """
                    session.run(create_query, dimensions=dimensions)
                    logger.info("Vector index created successfully")
                else:
                    logger.info("Vector index already exists")
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            raise

    def create_edge(self, edge_type: str, id1: str, id2: str, label1: str = "Memory", 
                   label2: str = "Memory", similarity: float = None, 
                   source: str = "auto", confidence: float = None):
        """
        Create an edge between two nodes in Neo4j.
        
        Args:
            edge_type: Type of the edge to create
            id1: ID of the source node
            id2: ID of the target node
            label1: Label of the source node
            label2: Label of the target node
            similarity: Optional similarity score
            source: Source of the edge creation
            confidence: Optional confidence score
        """
        try:
            with self.driver.session(database=self.db_name) as session:
                session.run(
                    f"""
                    MATCH (a:{label1} {{id: $id1}})
                    MATCH (b:{label2} {{id: $id2}})
                    MERGE (a)-[r:{edge_type}]->(b)
                    SET r.created_at = datetime(),
                        r.source = $source,
                        r.similarity = $similarity,
                        r.confidence = $confidence
                    """, 
                    id1=id1, id2=id2, similarity=similarity, 
                    source=source, confidence=confidence
                )
            logger.debug(f"Created {edge_type} edge from {id1} to {id2}")
        except Exception as e:
            logger.error(f"Error creating edge: {e}")

    def link_related_memory_nodes(self, node_id: str, similar_nodes: List[dict]):
        """
        Create RELATES_TO edges between a node and similar nodes.
        
        Args:
            node_id: ID of the source node
            similar_nodes: List of similar nodes with similarity scores
        """
        for match in similar_nodes:
            other_node = match["node"]
            similarity = match["similarity"]
            
            if other_node["id"] != node_id:
                if "Memory" in other_node.labels or "Knowledge" in other_node.labels:
                    try:
                        self.create_edge(
                            edge_type="RELATES_TO",
                            id1=node_id,
                            id2=other_node["id"],
                            similarity=similarity
                        )
                        logger.debug(
                            f"Linked {node_id} to {match['node']['id']} via RELATES_TO "
                            f"(similarity: {match['similarity']:.2f})"
                        )
                    except Exception as e:
                        logger.error(f"Error linking related nodes: {e}")

    def link_unsynthesized_memory_to_synthesis_nodes(self, synthesis_id: str, unsynthesized_nodes: List[dict]):
        for mem in unsynthesized_nodes:
            try:
                self.create_edge(
                    edge_type="SYNTHESIZED_BY",
                    id1=mem["id"],
                    id2=synthesis_id,
                    label1="Memory",
                    label2="Synthesis"
                )
                logger.debug(
                    f"memory node {mem['id']} to synthesis node {synthesis_id} via SYNTHESIZED_BY"
                )            
            except Exception as e:
                logger.error(f"Error linking related nodes: {e}")

    def link_extending_synthesis_nodes(self, synthesis_id: str, new_memory_id: str):
        """
        Create EXTENDS edges between a synthesis node and other synthesis or knowledge nodes
        found through memory node traversal.
        
        Args:
            synthesis_id: ID of the synthesis node
        """
        try:
            # Get all related memory node IDs from the cluster
            related_memory_ids = self.traverse_related_nodes([new_memory_id],start_labels=["Memory"],target_labels=["Memory"],edge_types=["RELATES_TO"])

            with self.driver.session(database=self.db_name) as session:
                # Find all synthesis nodes linked to those memory nodes
                result = session.run(
                    """
                    MATCH (m:Memory)-[:SYNTHESIZED_BY]->(s:Synthesis)
                    WHERE m.id IN $ids AND s.id <> $synthesis_id
                    RETURN DISTINCT s.id AS other_synthesis_id
                    """,
                    ids=related_memory_ids,
                    synthesis_id=synthesis_id
                )
                for record in result:
                    other_id = record["other_synthesis_id"]
                    self.create_edge(
                        edge_type="EXTENDS",
                        id1=synthesis_id,
                        id2=other_id,
                        label1="Synthesis",
                        label2="Synthesis"
                    )
                    logger.debug(f"Linked synthesis node {synthesis_id} to related synthesis node {other_id} via EXTENDS")
        except Exception as e:
            logger.error(f"Error linking extending synthesis nodes: {e}")

    def get_unsynthesized_related_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get related memory nodes that have not been synthesized.
        
        Args:
            node_ids: List of node IDs to check
            
        Returns:
            List of dictionaries with node ID and text
        """
        try:
            with self.driver.session(database=self.db_name) as session:
                result = session.run(
                    """
                    MATCH (m:Memory)
                    WHERE m.id IN $ids
                    AND NOT EXISTS {
                        MATCH (m)-[r]->()
                        WHERE type(r) = 'SYNTHESIZED_BY'
                    }
                    RETURN m.id AS mid,
                        m.prompt + ' ' + m.response AS text,
                        m.embedding AS embedding
                    """,
                    ids=node_ids
                )
                return [{"id": record["mid"], "text": record["text"], "embedding": record["embedding"]} for record in result]
        except Exception as e:
            logger.error(f"Error getting unsynthesized nodes: {e}")
            return []
        
    
    def traverse_related_nodes(
        self,
        node_ids: List[str],
        start_labels: Optional[List[str]] = None,
        target_labels: Optional[List[str]] = None,
        edge_types: Optional[List[str]] = None,
        max_depth: Optional[int] = None
    ) -> List[str]:
        """
        Traverse the graph to collect all connected nodes that match the given
        label / relationship filters.

        Args:
            node_ids:       List of starting node IDs.
            start_labels:   Optional labels that the *starting* node must have.
                            None / []  => no label filter.
            target_labels:  Optional labels that the *returned* nodes must have.
                            None / []  => no label filter.
            edge_types:     Optional relationship types to follow.
                            None / []  => follow any type.
            max_depth:     Optional maximum traversal depth (hop count).
                           None  => unbounded traversal.
        Returns:
            Distinct list of matched node IDs **including** the input IDs.
        """
        # If a parameter is None or an empty list => no filtering on that dimension
        if not start_labels:
            start_labels = []               # match any start node labels
        if not target_labels:
            target_labels = []              # match any target node labels
        if not edge_types:
            edge_types = []                 # follow any relationship types

        # Build APOC filter strings. Empty string == no restriction in APOC.
        edge_filter  = "|".join(edge_types) if edge_types else ""
        label_filter = "|".join(target_labels) if target_labels else ""

        # Depth handling
        depth_clause = ""
        if max_depth is not None and max_depth > 0:
            depth_clause = ", maxLevel: $max_level"

        try:
            with self.driver.session(database=self.db_name) as session:
                 # Build parameter map; include max_level only if depth was requested
                params = {
                    "ids": node_ids,
                    "start_labels": start_labels,
                    "target_labels": target_labels,
                    "edge_filter": edge_filter,
                    "label_filter": label_filter,
                }
                if max_depth is not None and max_depth > 0:
                    params["max_level"] = max_depth
               
                # Build the Cypher query with depth control injected
                cypher_query = f"""
                    // match starting nodes with optional label restriction
                    MATCH (start)
                    WHERE start.id IN $ids
                      AND (
                        size($start_labels) = 0
                        OR any(l IN labels(start) WHERE l IN $start_labels)
                      )

                    // traverse the subgraph using APOC
                    CALL apoc.path.subgraphNodes(
                      start,
                      {{
                        relationshipFilter: $edge_filter,
                        labelFilter: $label_filter{depth_clause}
                      }}
                    ) YIELD node

                    // optionally restrict the labels of returned nodes
                    WITH DISTINCT node
                    WHERE size($target_labels) = 0
                       OR any(l IN labels(node) WHERE l IN $target_labels)

                    RETURN DISTINCT node.id AS id
                """

                result = session.run(cypher_query, **params)
                return [record["id"] for record in result]
        except Exception as e:
            logger.error(f"Error traversing related nodes: {e}")
            return []

    def link_concept_to_node(self, concept_id: str, node_id: str):
        """
        Link a concept node to another node by creating a TAGS relationship.
        
        Args:
            concept_id: ID of the concept node
            node_id: ID of the target node
        """
        try:
            with self.driver.session(database=self.db_name) as session:
                session.run(
                    """
                    MATCH (c:Concept {id: $concept_id})
                    MATCH (n {id: $node_id})
                    MERGE (c)-[r:TAGS]->(n)
                    SET r.created_at = datetime()
                    """,
                    concept_id=concept_id,
                    node_id=node_id
                )
                logger.debug(f"Created TAGS edge from Concept {concept_id} to node {node_id}")
        except Exception as e:
            logger.error(f"Error linking concept to node: {e}")
            raise

    def fetch_edges_between_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch all edges between the specified nodes.

        Args:
            node_ids: List of node IDs to check for relationships

        Returns:
            List of edges with type, source, target, and optional properties
        """
        try:
            with self.driver.session(database=self.db_name) as session:
                result = session.run(
                    """
                    MATCH (a)-[r]->(b)
                    WHERE a.id IN $ids AND b.id IN $ids
                    RETURN a.id AS source, b.id AS target, type(r) AS type, r AS properties
                    """,
                    ids=node_ids
                )
                return [
                    {
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["type"],
                        "properties": dict(record["properties"])
                    } for record in result
                ]
        except Exception as e:
            logger.error(f"Error fetching edges between nodes: {e}")
            return []
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a node by its ID.

        Args:
            node_id: ID of the node to fetch

        Returns:
            Dictionary containing node data, labels, and ID or None if not found
        """
        try:
            with self.driver.session(database=self.db_name) as session:
                result = session.run(
                    """
                    MATCH (n)
                    WHERE n.id = $node_id
                    RETURN n AS node, labels(n) AS labels, n.id AS id
                    """,
                    node_id=node_id
                )
                record = result.single()
                if record:
                    return {
                        "source": "neo4j",
                        "node": record["node"],
                        "labels": record["labels"],
                        "id": record["id"]
                    }
                return None
        except Exception as e:
            logger.error(f"Error fetching node by ID {node_id}: {e}")
            return None

    def get_sources(self) -> List[str]:
        """
        Get all unique source names from Knowledge nodes in the database.
        
        Returns:
            List of unique source names
        """
        try:
            with self.driver.session(database=self.db_name) as session:
                result = session.run(
                    """
                    MATCH (k:Knowledge)
                    RETURN DISTINCT k.source_name AS source
                    ORDER BY source
                    """
                )
                sources = [record["source"] for record in result]
                logger.debug(f"Retrieved sources from knowledge nodes: {sources}")
                return sources
        except Exception as e:
            logger.error(f"Error fetching knowledge sources: {e}")
            return []
    
    def delete_source(self, source_name: str) -> bool:
        """
        Delete all Knowledge nodes with the specified source name and perform cleanup:
        1. Delete all relationships involving these Knowledge nodes
        2. Delete any Concept nodes that would become orphaned (no remaining connections)
        
        Args:
            source_name: The name of the source to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session(database=self.db_name) as session:
                # First, identify all Knowledge nodes that will be deleted
                knowledge_ids_result = session.run(
                    """
                    MATCH (k:Knowledge {source_name: $source_name})
                    RETURN k.id AS id
                    """,
                    source_name=source_name
                )
                knowledge_ids = [record["id"] for record in knowledge_ids_result]
                logger.debug(f"There are {len(knowledge_ids)} Knowledge nodes for deletion in source: {source_name}")
                
                # Identify Concept nodes that will become orphaned after deletion
                # These are Concept nodes that ONLY connect to the Knowledge nodes we're about to delete
                orphaned_concepts_result = session.run(
                    """
                    // Find concepts connected to nodes in this source
                    MATCH (c:Concept)-[:TAGS]->(k:Knowledge {source_name: $source_name})
                    WITH c, count(k) as connections_to_source
                    
                    // Count connections to ANY node
                    MATCH (c)-[:TAGS]->(any)
                    WITH c, connections_to_source, count(any) as total_connections
                    
                    // If there are Concept nodes that are linked to nodes in this source, the concept nodes will become orphaned and thus needs to be deleted
                    WHERE connections_to_source = total_connections
                    RETURN c.id AS id, c.phrase AS phrase
                    """,
                    source_name=source_name
                )
                orphaned_concept_ids = [record["id"] for record in orphaned_concepts_result]
                logger.debug(f"There are {len(orphaned_concept_ids)} orphaned Concept nodes that will be deleted")
                
                # Now perform the deletion in the correct order:
                
                # 1. Delete all relationships involving Knowledge nodes from this source
                # This avoids constraint violations when deleting the nodes
                session.run(
                    """
                    MATCH (k:Knowledge {source_name: $source_name})-[r]-()
                    DELETE r
                    """,
                    source_name=source_name
                )
                logger.debug(f"Deleted all relationships for Knowledge nodes from source: {source_name}")
                
                # 2. Delete all Knowledge nodes from this source
                knowledge_delete_result = session.run(
                    """
                    MATCH (k:Knowledge {source_name: $source_name})
                    DELETE k
                    """,
                    source_name=source_name
                )
                logger.debug(f"Deleted all Knowledge nodes for source: {source_name}")
                
                # 3. Delete any orphaned Concept nodes
                if orphaned_concept_ids:
                    # First delete their relationships
                    session.run(
                        """
                        MATCH (c:Concept)-[r]-()
                        WHERE c.id IN $orphaned_ids
                        DELETE r
                        """,
                        orphaned_ids=orphaned_concept_ids
                    )
                    
                    # Then delete the nodes themselves
                    concept_delete_result = session.run(
                        """
                        MATCH (c:Concept)
                        WHERE c.id IN $orphaned_ids
                        DELETE c
                        """,
                        orphaned_ids=orphaned_concept_ids
                    )
                    logger.debug(f"Deleted {len(orphaned_concept_ids)} orphaned Concept nodes")
                
                return True
        except Exception as e:
            logger.error(f"Error deleting source {source_name}: {e}")
            return False

def get_db_client(config = None, vector_index_dimentions: int = 384) -> BaseDBClient:
    """
    Factory function to return a configured database client based on config.
    Currently only supports Neo4j.
    
    Args:
        config: Optional Config instance. If not provided, will create a new one.
        
    Returns:
        Configured database client
        
    Raises:
        ValueError: If the configured provider is not supported
    """
    from config import Config
    if config is None:
        config = Config()
        
    provider = config.get('db', 'provider')
    db_name = config.get('db', 'name')
    
    if provider == 'neo4j':
        neo4j_config = config.get('db', 'neo4j')
        return Neo4jManager(
            host=neo4j_config['host'],
            port=neo4j_config['port'],
            user=neo4j_config['user'],
            password=neo4j_config['password'],
            db_name=db_name,

        )
    else:
        raise ValueError(f"Unsupported database provider: {provider}")
