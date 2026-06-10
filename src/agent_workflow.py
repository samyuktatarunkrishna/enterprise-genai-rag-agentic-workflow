"""
Agentic Workflow module.
Implements a 4-stage LangGraph-inspired agentic RAG workflow.
"""
import time
from typing import List, Dict, Any, Tuple
from src.utils import clean_text
from src.retriever import Retriever
from src.tools import ToolRouter
from src.rag_pipeline import RAGPipeline

class AgentWorkflow:
    """
    Executes a structured RAG query answering workflow through 4 stages:
    1. Query Handling (cleaning and categorization)
    2. Retrieval (semantic vector search)
    3. Tool Routing (metadata stats vs. document lookup)
    4. Controlled Response Generation (citations, confidence, and output structuring)
    """
    def __init__(self, retriever: Retriever, tool_router: ToolRouter, rag_pipeline: RAGPipeline):
        """
        Initializes the AgentWorkflow.
        
        Args:
            retriever (Retriever): Retriever instance.
            tool_router (ToolRouter): ToolRouter instance.
            rag_pipeline (RAGPipeline): RAGPipeline instance.
        """
        self.retriever = retriever
        self.tool_router = tool_router
        self.rag_pipeline = rag_pipeline

    def _stage_query_handling(self, query: str) -> Tuple[str, str]:
        """
        Stage 1: Clean and classify the incoming query.
        """
        cleaned_query = clean_text(query)
        
        # Keyword mappings to classify the query type
        cleaned_lower = cleaned_query.lower()
        
        # Check metadata keywords first
        if any(w in cleaned_lower for w in ["statistics", "count", "metadata", "how many documents", "stats", "how many"]):
            query_type = "metadata"
        elif any(w in cleaned_lower for w in ["remote work", "leave", "onboarding", "benefits", "pto", "holiday", "sick", "employee"]):
            query_type = "policy"
        elif any(w in cleaned_lower for w in ["password", "vpn", "laptop", "access", "wifi", "reset", "login", "mfa"]):
            query_type = "IT support"
        elif any(w in cleaned_lower for w in ["pipeline", "crm", "lead", "approval", "discount", "deal", "bant", "mql", "sql"]):
            query_type = "sales"
        elif any(w in cleaned_lower for w in ["product", "pricing", "support options", "tier", "features", "use cases", "suite"]):
            query_type = "product"
        elif any(w in cleaned_lower for w in ["privacy", "approved tool", "escalation", "audit", "compliance", "gdpr", "ccpa", "pii", "ciso"]):
            query_type = "compliance"
        else:
            query_type = "unknown"
            
        return cleaned_query, query_type

    def _stage_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """
        Stage 2: Retrieve relevant chunks.
        """
        # Retrieve chunks using config-defined settings
        return self.retriever.retrieve(query)

    def _stage_tool_routing(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Stage 3: Decide and call the matching MCP tool.
        """
        return self.tool_router.route_and_execute(query)

    def _stage_response_generation(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        selected_tool: str, 
        tool_output: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Stage 4: Generate a cited response based on retrieval and tools.
        """
        if selected_tool == "metadata_statistics_tool":
            # Generate a structured answer summarizing the corpus metadata
            docs_list = ", ".join(tool_output["available_documents"])
            answer = (
                f"Based on the system diagnostics tool (metadata_statistics_tool):\n"
                f"- Total Available Documents: {tool_output['num_documents']} ({docs_list})\n"
                f"- Total Ingested Text Chunks: {tool_output['num_chunks']}\n"
                f"- Average Chunk Length: {tool_output['average_chunk_length']} characters\n\n"
                f"[References: System Metadata Service]"
            )
            confidence = "High"
        else:
            # Generate an answer from the retrieved text context
            answer, confidence = self.rag_pipeline.generate_response(query, retrieved_chunks)
            
        return answer, confidence

    def run(self, query: str) -> Dict[str, Any]:
        """
        Orchestrates the 4-stage agentic workflow.
        
        Args:
            query (str): User query.
            
        Returns:
            Dict[str, Any]: Formatted execution results.
        """
        start_time = time.perf_counter()
        
        # Stage 1: Query Handling
        cleaned_query, query_type = self._stage_query_handling(query)
        
        # Stage 2: Retrieval
        retrieved_context = self._stage_retrieval(cleaned_query)
        
        # Stage 3: Tool Routing
        selected_tool, tool_output = self._stage_tool_routing(cleaned_query)
        
        # Stage 4: Response Generation
        answer, confidence = self._stage_response_generation(
            cleaned_query, 
            retrieved_context, 
            selected_tool, 
            tool_output
        )
        
        latency = time.perf_counter() - start_time
        
        return {
            "query": query,
            "query_type": query_type,
            "selected_tool": selected_tool,
            "retrieved_context": retrieved_context,
            "answer": answer,
            "confidence": confidence,
            "latency_seconds": round(latency, 5)
        }
