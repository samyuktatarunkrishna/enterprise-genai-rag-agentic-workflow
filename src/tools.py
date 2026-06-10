"""
MCP-style Tools and Routing module.
Provides diagnostic tools and dynamic tool routing based on query intent.
"""
from typing import List, Dict, Any, Tuple
from src.retriever import Retriever

def document_lookup_tool(query: str, retriever: Retriever) -> Dict[str, Any]:
    """
    Simulates lookup tool that fetches matching document chunks and details.
    
    Args:
        query (str): Search query.
        retriever (Retriever): Retriever instance.
        
    Returns:
        Dict[str, Any]: Tool results including count and matched chunks.
    """
    retrieved_results = retriever.retrieve(query)
    return {
        "tool_name": "document_lookup_tool",
        "query": query,
        "matches_found": len(retrieved_results),
        "results": retrieved_results
    }

def metadata_statistics_tool(chunks: List[Dict[str, Any]], documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simulates diagnostic tool returning general metadata statistics.
    
    Args:
        chunks (List[Dict[str, Any]]): List of all document chunks.
        documents (List[Dict[str, Any]]): List of all loaded documents.
        
    Returns:
        Dict[str, Any]: Descriptive corpus statistics.
    """
    num_docs = len(documents)
    num_chunks = len(chunks)
    avg_len = sum(len(c["chunk_text"]) for c in chunks) / num_chunks if num_chunks > 0 else 0.0
    doc_names = sorted(list({c["document_name"] for c in chunks}))
    
    return {
        "tool_name": "metadata_statistics_tool",
        "num_documents": num_docs,
        "num_chunks": num_chunks,
        "average_chunk_length": round(avg_len, 2),
        "available_documents": doc_names
    }

class ToolRouter:
    """
    Analyzes queries and routes them to the correct MCP tool.
    """
    def __init__(self, retriever: Retriever, chunks: List[Dict[str, Any]], documents: List[Dict[str, Any]]):
        """
        Initializes the ToolRouter.
        
        Args:
            retriever (Retriever): The initialized retriever instance.
            chunks (List[Dict[str, Any]]): Complete list of chunks.
            documents (List[Dict[str, Any]]): Complete list of raw documents.
        """
        self.retriever = retriever
        self.chunks = chunks
        self.documents = documents

    def route_and_execute(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Decides which tool to invoke based on keyword analysis, then runs it.
        
        Args:
            query (str): The search query.
            
        Returns:
            Tuple[str, Dict[str, Any]]: (selected_tool_name, tool_output)
        """
        cleaned_query = query.lower()
        stats_keywords = {"statistics", "count", "metadata", "documents", "stats", "average", "how many"}
        
        # Determine if the query is asking about the corpus metadata
        if any(keyword in cleaned_query for keyword in stats_keywords):
            tool_name = "metadata_statistics_tool"
            tool_output = metadata_statistics_tool(self.chunks, self.documents)
        else:
            tool_name = "document_lookup_tool"
            tool_output = document_lookup_tool(query, self.retriever)
            
        return tool_name, tool_output
