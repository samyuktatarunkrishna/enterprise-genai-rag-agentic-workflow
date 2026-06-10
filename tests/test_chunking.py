"""
Unit tests for the document chunking module.
"""
import pytest
from src.chunking import TextChunker

def test_split_documents_into_chunks():
    """
    Tests that a typical document is split into overlapping chunks,
    and metadata properties (such as indices and character bounds) are accurately preserved.
    """
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    doc = {
        "document_id": "test_doc_1",
        "document_name": "hr_policy.txt",
        "text": "This is a sample text document designed to test the chunking process. It should create multiple chunks.",
        "metadata": {"department": "HR"}
    }
    
    chunks = chunker.split_document(doc)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["document_id"] == "test_doc_1"
        assert chunk["document_name"] == "hr_policy.txt"
        assert len(chunk["chunk_text"]) > 0
        assert chunk["chunk_id"].startswith("test_doc_1_c")
        assert "start_char" in chunk["metadata"]
        assert "end_char" in chunk["metadata"]
        assert chunk["metadata"]["department"] == "HR"

def test_prevent_empty_chunks():
    """
    Tests that documents containing only whitespace or empty text yield zero chunks.
    """
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    empty_doc = {
        "document_id": "empty_doc",
        "document_name": "empty.txt",
        "text": "   \n\n   ",
        "metadata": {}
    }
    
    chunks = chunker.split_document(empty_doc)
    assert len(chunks) == 0
