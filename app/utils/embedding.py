"""
Service for generating embeddings and NLP operations.
"""
import nltk
from typing import List
import numpy as np
from config import Config
from utils.logging_utils import setup_logging
import tiktoken                      

# tokenizer for counting tokens
ENC = tiktoken.get_encoding("cl100k_base")   # good default for Mistral

# Ensure the tokenizer models are available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

#load config and setup logging
config = Config()
logger = setup_logging(config, __name__)

# Initialize models
try:
    from sentence_transformers import SentenceTransformer
    config = Config()
    embedding_model = SentenceTransformer(config.get("embeddings", "model"))
    logger.info("Loaded SentenceTransformer embedding model")
except ImportError as e:
    logger.error(f"Failed to load SentenceTransformer: {e}")
    embedding_model = None

try:
    from transformers import pipeline
    summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    logger.info("Loaded transformers summarization pipeline")
except ImportError as e:
    logger.error(f"Failed to load transformers pipeline: {e}")
    summarizer_pipeline = None

def generate_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector from input text.
    
    Args:
        text: Text to embed
        
    Returns:
        List of embedding values
        
    Raises:
        RuntimeError: If the embedding model is not available
    """
    if embedding_model is None:
        raise RuntimeError("Embedding model is not available")
    
    try:
        embedding = embedding_model.encode(text, show_progress_bar=False)
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

def generate_mixed_embedding(first_embedding: list, second_embedding: list, first_embedding_weight=0.5):
    """
    Creates a contextualized concept embedding by blending concept and the parent node embeddings,
    then normalizes the result for similarity search.

    Args:
        concept_embedding (list): Embedding of the concept node
        parent_node_embedding (list): Embedding of the parent node
        concept_weight (float): Weight for the concept embedding (0.0–1.0)

    Returns:
        list: Normalized contextualized embedding
    """
    try:
        # Convert lists to numpy arrays
        first_emb = np.array(first_embedding)
        second_emb = np.array(second_embedding)
        
        # Perform the weighted sum
        vec = first_embedding_weight * first_emb + (1 - first_embedding_weight) * second_emb
        norm = np.linalg.norm(vec)
        
        # Normalize and convert back to list
        normalized_vec = vec / (norm + 1e-8)
        return normalized_vec.tolist()
    except Exception as e:
        logger.error(f"Error in contextualize_concept: {e}")
        raise

def count_tokens(text: str) -> int:
    return len(ENC.encode(text))