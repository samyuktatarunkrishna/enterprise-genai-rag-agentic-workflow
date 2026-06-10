"""
Configuration file for the Enterprise GenAI RAG and Agentic Workflow System.
"""

# Text chunking configuration
CHUNK_SIZE = 450      # Number of characters per chunk
CHUNK_OVERLAP = 80    # Overlap characters between consecutive chunks

# Retrieval configuration
TOP_K = 3             # Number of top relevant chunks to retrieve
SCORE_THRESHOLD = 0.08 # Minimum similarity score threshold for relevance
