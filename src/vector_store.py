"""
In-memory vector database.
Stores chunk definitions and conducts similarity searches using cosine similarity.
"""
from typing import List, Dict, Any, Tuple
import numpy as np

class InMemoryVectorStore:
    """
    Lightweight, in-memory vector database representation using numpy.
    """
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.vectors: np.ndarray = np.empty((0, 0))

    def add_chunks(self, chunks: List[Dict[str, Any]], vectors: np.ndarray) -> None:
        """
        Adds text chunks and their corresponding embedding vectors to the store.
        
        Args:
            chunks (List[Dict[str, Any]]): List of chunk dictionaries.
            vectors (np.ndarray): Dense 2D numpy array containing corresponding embedding vectors.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: Got {len(chunks)} chunks, but {len(vectors)} vectors."
            )
            
        self.chunks.extend(chunks)
        
        if self.vectors.size == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def similarity_search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Calculates cosine similarity between the query vector and stored vectors.
        
        Args:
            query_vector (np.ndarray): Dense 1D vector of the search query.
            top_k (int): Number of top results to return.
            
        Returns:
            List[Tuple[Dict[str, Any], float]]: Top K matching tuples of (chunk_dict, cosine_similarity_score).
        """
        if len(self.chunks) == 0 or self.vectors.size == 0:
            return []
            
        # Cosine similarity: dot(A, B) / (norm(A) * norm(B))
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            # Query is empty or has no words matching vocab. Return first top_k with 0 score
            return [(chunk, 0.0) for chunk in self.chunks[:top_k]]
            
        # Compute norms along rows of matrix
        norms = np.linalg.norm(self.vectors, axis=1)
        # Prevent division by zero
        norms[norms == 0] = 1.0
        
        # Vectorized dot product
        dot_products = np.dot(self.vectors, query_vector)
        
        # Cosine similarity calculations
        scores = dot_products / (norms * query_norm)
        
        # Find indices of sorted elements in descending order
        sorted_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append((self.chunks[idx], float(scores[idx])))
            
        return results
