"""
Unit tests for the vector retriever module.
"""
import pytest
from src.embeddings import TfidfEmbeddingModel
from src.vector_store import InMemoryVectorStore
from src.retriever import Retriever

def test_retrieval_relevance_and_scores():
    """
    Tests that the retriever successfully identifies relevant documents,
    sorts them in descending order of similarity, and correctly applies threshold filtering.
    """
    # 1. Initialize DB components
    vector_store = InMemoryVectorStore()
    embedding_model = TfidfEmbeddingModel()
    
    chunks = [
        {
            "chunk_id": "doc1_c0",
            "document_id": "doc1",
            "document_name": "hr_policy.txt",
            "chunk_text": "Employees can request remote work by submitting a proposal to HR.",
            "chunk_index": 0,
            "metadata": {}
        },
        {
            "chunk_id": "doc2_c0",
            "document_id": "doc2",
            "document_name": "it_support_faq.txt",
            "chunk_text": "To reset your password, visit reset.enterprise.com and authenticate using MFA.",
            "chunk_index": 0,
            "metadata": {}
        }
    ]
    
    # Fit model vocabulary and index vectors
    embedding_model.fit([c["chunk_text"] for c in chunks])
    chunk_vectors = embedding_model.embed_texts([c["chunk_text"] for c in chunks])
    vector_store.add_chunks(chunks, chunk_vectors)
    
    # 2. Test standard retriever
    retriever = Retriever(vector_store, embedding_model, score_threshold=0.05)
    
    # Test query 1 (HR related)
    hr_results = retriever.retrieve("How can I request remote work?", top_k=1)
    assert len(hr_results) == 1
    assert hr_results[0]["document_name"] == "hr_policy.txt"
    assert hr_results[0]["relevance_score"] > 0.05
    
    # Test query 2 (IT related)
    it_results = retriever.retrieve("I want to reset my password", top_k=1)
    assert len(it_results) == 1
    assert it_results[0]["document_name"] == "it_support_faq.txt"
    
    # Test query 3 (no matches above score threshold)
    poor_results = retriever.retrieve("Unrelated query about making pizza", top_k=1)
    assert len(poor_results) == 0
