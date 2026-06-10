"""
Semantic Retriever module.
Orchestrates embedding generation, vector store querying, and score threshold filtering.
"""
from typing import List, Dict, Any
from src.embeddings import TfidfEmbeddingModel
from src.vector_store import InMemoryVectorStore

class Retriever:
    """
    Interface for semantic retrieval, mapping queries to matching chunks.
    """
    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        embedding_model: TfidfEmbeddingModel,
        score_threshold: float
    ):
        """
        Initializes the Retriever.
        
        Args:
            vector_store (InMemoryVectorStore): Vector database instance.
            embedding_model (TfidfEmbeddingModel): Text embedder.
            score_threshold (float): Minimum score to include in results.
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.score_threshold = score_threshold

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves relevant document chunks sorted by similarity score, filtered by threshold.
        
        Args:
            query (str): Input query from user.
            top_k (int): Number of top documents to fetch.
            
        Returns:
            List[Dict[str, Any]]: Clean list of retrieved chunk dictionaries.
            Each dictionary contains:
                - chunk_id (str)
                - document_name (str)
                - chunk_text (str)
                - relevance_score (float)
                - metadata (dict)
        """
        if not query or not query.strip():
            return []
            
        # Get query vector representation
        query_vector = self.embedding_model.embed_query(query)
        
        # Search in the database
        raw_results = self.vector_store.similarity_search(query_vector, top_k=top_k)
        
        # Filter results using the configuration threshold
        retrieved_results = []
        for chunk, score in raw_results:
            if score >= self.score_threshold:
                retrieved_results.append({
                    "chunk_id": chunk["chunk_id"],
                    "document_name": chunk["document_name"],
                    "chunk_text": chunk["chunk_text"],
                    "relevance_score": score,
                    "metadata": chunk["metadata"]
                })
                
        return retrieved_results
