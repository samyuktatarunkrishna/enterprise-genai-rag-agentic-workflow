"""
Controlled RAG Response Generation module.
Synthesizes answers from retrieved chunks without hallucination.
"""
import re
from typing import List, Dict, Any, Tuple

class RAGPipeline:
    """
    Controlled response generator that structures answers using only retrieved context.
    """
    def __init__(self, fallback_message: str = None):
        """
        Initializes the RAGPipeline.
        
        Args:
            fallback_message (str): Message to return when context is insufficient.
        """
        self.fallback_message = fallback_message or (
            "I do not have enough information in the available documents to answer this confidently."
        )

    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Generates a RAG-based response using retrieved document sentences matching query keywords.
        
        Args:
            query (str): The search query.
            retrieved_chunks (List[Dict[str, Any]]): Top retrieved chunks.
            
        Returns:
            Tuple[str, str]: (generated_answer, confidence_level)
        """
        if not retrieved_chunks:
            return self.fallback_message, "Low"
            
        # Extract query terms, ignoring common retrieval stopwords
        stopwords = {
            "what", "how", "why", "who", "which", "where", "when", "the", "a", "an", 
            "is", "are", "do", "does", "should", "can", "to", "for", "in", "on", "at", 
            "of", "and", "or", "about", "process", "request", "available", "tier", 
            "suitable", "support", "options", "escalated", "get", "need"
        }
        
        # Normalize and filter terms
        query_words = [w.lower().strip("?,.!") for w in query.split()]
        query_keywords = {w for w in query_words if w and w not in stopwords}
        
        # Calculate maximum relevance score
        max_score = max(chunk["relevance_score"] for chunk in retrieved_chunks)
        
        # Determine confidence metrics
        if max_score >= 0.35:
            confidence = "High"
        elif max_score >= 0.20:
            confidence = "Medium"
        else:
            confidence = "Low"
            
        findings = []
        cited_docs = set()
        
        # Perform sentence extraction on each retrieved chunk
        for chunk in retrieved_chunks:
            doc_name = chunk["document_name"]
            text = chunk["chunk_text"]
            
            # Split sentences using basic lookbehind regex for sentence endings
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            chunk_findings = []
            for sentence in sentences:
                sentence_clean = sentence.lower()
                # Count keyword matches
                score = sum(1 for kw in query_keywords if kw in sentence_clean)
                if score > 0:
                    chunk_findings.append((score, sentence))
            
            # Sort findings by keyword match relevance
            chunk_findings.sort(key=lambda x: x[0], reverse=True)
            
            if chunk_findings:
                # Take up to the top 2 matching sentences
                selected = [item[1] for item in chunk_findings[:2]]
                combined = " ".join(selected)
                findings.append(f"- {combined} (Source: {doc_name})")
                cited_docs.add(doc_name)
            elif sentences:
                # Fallback: take the first sentence of the chunk if no keyword hits occur
                findings.append(f"- {sentences[0]} (Source: {doc_name})")
                cited_docs.add(doc_name)
                
        if not findings:
            return self.fallback_message, "Low"
            
        # Format a professional markdown response
        intro = "Based on the retrieved enterprise documentation, here is what we found:\n\n"
        body = "\n".join(findings)
        
        citations = ", ".join(sorted(list(cited_docs)))
        outro = f"\n\n[References: {citations}]"
        
        answer = intro + body + outro
        return answer, confidence
