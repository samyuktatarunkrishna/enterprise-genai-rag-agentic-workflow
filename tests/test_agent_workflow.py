"""
Unit tests for the agent workflow module.
"""
import pytest
from src.embeddings import TfidfEmbeddingModel
from src.vector_store import InMemoryVectorStore
from src.retriever import Retriever
from src.tools import ToolRouter
from src.rag_pipeline import RAGPipeline
from src.agent_workflow import AgentWorkflow

def test_agent_workflow_execution_and_structure():
    """
    Tests that the 4-stage agentic workflow processes lookup and diagnostic queries,
    routes them to the correct tools, and outputs a complete results schema.
    """
    # 1. Setup mock resources
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
        }
    ]
    documents = [
        {
            "document_id": "doc1",
            "document_name": "hr_policy.txt",
            "text": "Employees can request remote work by submitting a proposal to HR.",
            "metadata": {}
        }
    ]
    
    # Ingest and fit
    embedding_model.fit([c["chunk_text"] for c in chunks])
    chunk_vectors = embedding_model.embed_texts([c["chunk_text"] for c in chunks])
    vector_store.add_chunks(chunks, chunk_vectors)
    
    retriever = Retriever(vector_store, embedding_model, score_threshold=0.01)
    tool_router = ToolRouter(retriever, chunks, documents)
    rag_pipeline = RAGPipeline()
    
    # Initialize Agent
    workflow = AgentWorkflow(retriever, tool_router, rag_pipeline)
    
    # 2. Test Document Lookup flow
    result = workflow.run("How do I request remote work?")
    
    # Verify exact schema keys exist
    assert "query" in result
    assert "query_type" in result
    assert "selected_tool" in result
    assert "retrieved_context" in result
    assert "answer" in result
    assert "confidence" in result
    assert "latency_seconds" in result
    
    # Validate content properties
    assert result["query_type"] == "policy"
    assert result["selected_tool"] == "document_lookup_tool"
    assert len(result["retrieved_context"]) == 1
    assert "hr_policy.txt" in result["answer"]
    assert result["latency_seconds"] > 0
    
    # 3. Test Metadata Statistics flow
    meta_result = workflow.run("Show metadata statistics.")
    assert meta_result["query_type"] == "metadata"
    assert meta_result["selected_tool"] == "metadata_statistics_tool"
    assert "Total Available Documents" in meta_result["answer"]
    assert meta_result["confidence"] == "High"
