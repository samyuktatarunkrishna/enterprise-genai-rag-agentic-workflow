"""
Unit tests for the RAG pipeline module.
"""
import pytest
from src.rag_pipeline import RAGPipeline

def test_rag_fallback_on_weak_context():
    """
    Tests that the RAG pipeline outputs the standard fallback statement
    and assigns a Low confidence rating when context is absent or weak.
    """
    pipeline = RAGPipeline()
    
    # Perform generation with no context
    answer, confidence = pipeline.generate_response(
        query="What is the corporate policy on alien invasions?",
        retrieved_chunks=[]
    )
    
    expected_fallback = "I do not have enough information in the available documents to answer this confidently."
    assert answer == expected_fallback
    assert confidence == "Low"

def test_rag_successful_synthesis():
    """
    Tests that relevant retrieved context produces a formatted, cited response
    and maps the confidence correctly according to the matching score.
    """
    pipeline = RAGPipeline()
    
    retrieved_context = [
        {
            "chunk_id": "hr_1",
            "document_name": "hr_policy.txt",
            "chunk_text": "Employees can request remote work by submitting a proposal to their direct manager. Core hours are 10 AM to 4 PM.",
            "relevance_score": 0.52,
            "metadata": {}
        }
    ]
    
    answer, confidence = pipeline.generate_response(
        query="How to request remote work?",
        retrieved_chunks=retrieved_context
    )
    
    # Check that answer includes extracted information and source metadata citations
    assert "remote work" in answer.lower()
    assert "hr_policy.txt" in answer
    assert "[References: hr_policy.txt]" in answer
    assert confidence == "High"
