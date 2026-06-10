"""
Evaluation module for assessing RAG system performance.
Evaluates queries across relevance, hallucination risk, latency, and prompt metrics.
"""
import time
import re
from typing import List, Dict, Any
from src.agent_workflow import AgentWorkflow

# Define the 10 standard evaluation queries with expected targets
EVAL_QUERIES = [
    {
        "query": "How can an employee request remote work?",
        "expected_doc": "hr_policy.txt",
        "expected_type": "policy"
    },
    {
        "query": "What should I do if I cannot access VPN?",
        "expected_doc": "it_support_faq.txt",
        "expected_type": "IT support"
    },
    {
        "query": "What are the sales pipeline stages?",
        "expected_doc": "sales_process.txt",
        "expected_type": "sales"
    },
    {
        "query": "Which product tier is suitable for enterprise customers?",
        "expected_doc": "product_knowledge.txt",
        "expected_type": "product"
    },
    {
        "query": "What are the data privacy requirements?",
        "expected_doc": "compliance_guidelines.txt",
        "expected_type": "compliance"
    },
    {
        "query": "How many documents are available?",
        "expected_doc": "metadata",  # expecting metadata statistics tool
        "expected_type": "metadata"
    },
    {
        "query": "Show metadata statistics.",
        "expected_doc": "metadata",
        "expected_type": "metadata"
    },
    {
        "query": "What is the process for password reset?",
        "expected_doc": "it_support_faq.txt",
        "expected_type": "IT support"
    },
    {
        "query": "When should compliance issues be escalated?",
        "expected_doc": "compliance_guidelines.txt",
        "expected_type": "compliance"
    },
    {
        "query": "What support options are available for customers?",
        "expected_doc": "product_knowledge.txt",
        "expected_type": "product"
    }
]

def evaluate_hallucination_risk(answer: str, query: str, context: List[Dict[str, Any]], selected_tool: str) -> bool:
    """
    Checks if the generated answer contains words not found in either the query or the context.
    Provides a simple, transparent check of response integrity.
    """
    if selected_tool == "metadata_statistics_tool":
        # Checked programmatically, safe from textual hallucination
        return False
        
    if "I do not have enough information" in answer:
        # Fallback response is safe
        return False

    # Extract alphanumeric words longer than 3 characters, ignoring case
    answer_words = set(re.findall(r'[a-zA-Z]{4,}', answer.lower()))
    
    # Compile words from context sources and original query
    context_text = " ".join([c["chunk_text"] for c in context])
    source_text = (query + " " + context_text).lower()
    source_words = set(re.findall(r'[a-zA-Z]{4,}', source_text))
    
    # Common words in formatting templates to exclude from risk checking
    ignore_formatting = {
        "based", "retrieved", "enterprise", "documentation", "here", "found", 
        "references", "reference", "source", "according", "following", "under", 
        "bullet", "section", "chapter", "paragraph", "document", "documents",
        "what", "we", "is", "are", "here", "how", "why", "who", "which", "where"
    }
    
    # Dynamically extract and ignore words from retrieved source document names (e.g. 'hr', 'policy', 'txt')
    for chunk in context:
        doc_name_words = re.findall(r'[a-zA-Z]{2,}', chunk["document_name"].lower())
        ignore_formatting.update(doc_name_words)
        
    # Filter out formatting words
    answer_words = answer_words - ignore_formatting
    
    # Identify unique words generated in the answer not in the sources
    hallucinated_words = answer_words - source_words
    
    # Flag as high risk if more than 2 non-source words are found
    return len(hallucinated_words) > 2

def evaluate_prompt_quality(query: str, answer: str) -> float:
    """
    Evaluates prompt layout based on standard rules (instruction, structure, citations).
    Returns a score between 0.0 and 1.0.
    """
    score = 0.0
    # Did the system structure the answer with bullet points? (typical format)
    if "-" in answer:
        score += 0.3
    # Did it provide explicit citations/references?
    if "[References:" in answer or "Source:" in answer:
        score += 0.4
    # Did it generate a valid response (not empty)?
    if len(answer) > 50:
        score += 0.3
    return round(score, 2)

def run_evaluation(agent_workflow: AgentWorkflow) -> Dict[str, Any]:
    """
    Runs evaluation against the EVAL_QUERIES suite using the agent workflow.
    """
    print("\n" + "="*80)
    print("                      AGENTIC RAG SYSTEM EVALUATION RUN")
    print("="*80)
    
    results = []
    total_latency = 0.0
    total_relevance = 0.0
    hallucination_count = 0
    total_prompt_quality = 0.0
    
    print(f"{'Query':<50} | {'Type':<10} | {'Rel Score':<9} | {'Halluc Risk':<11} | {'Latency':<8}")
    print("-"*97)
    
    for item in EVAL_QUERIES:
        query = item["query"]
        expected_doc = item["expected_doc"]
        expected_type = item["expected_type"]
        
        # Execute workflow
        res = agent_workflow.run(query)
        
        latency = res["latency_seconds"]
        retrieved = res["retrieved_context"]
        answer = res["answer"]
        selected_tool = res["selected_tool"]
        
        # 1. Retrieval Relevance Score calculation
        # If query is metadata, checks if the statistics tool was correctly routed
        if expected_doc == "metadata":
            rel_score = 1.0 if selected_tool == "metadata_statistics_tool" else 0.0
        else:
            # Check if expected document name matches any retrieved chunk
            if retrieved:
                matching_chunks = sum(1 for c in retrieved if c["document_name"] == expected_doc)
                rel_score = matching_chunks / len(retrieved)
            else:
                rel_score = 0.0
                
        # 2. Hallucination Risk assessment
        hallucination_risk = evaluate_hallucination_risk(answer, query, retrieved, selected_tool)
        
        # 3. Prompt Quality score
        prompt_quality = evaluate_prompt_quality(query, answer)
        
        results.append({
            "query": query,
            "query_type": res["query_type"],
            "expected_type": expected_type,
            "selected_tool": selected_tool,
            "relevance_score": rel_score,
            "hallucination_risk": hallucination_risk,
            "latency": latency,
            "prompt_quality": prompt_quality,
            "confidence": res["confidence"]
        })
        
        total_latency += latency
        total_relevance += rel_score
        if hallucination_risk:
            hallucination_count += 1
        total_prompt_quality += prompt_quality
        
        # Print row
        q_trunc = query[:47] + "..." if len(query) > 50 else query
        print(f"{q_trunc:<50} | {res['query_type']:<10} | {rel_score:<9.2f} | {str(hallucination_risk):<11} | {latency:<8.4f}s")
        
    num_queries = len(EVAL_QUERIES)
    avg_latency = total_latency / num_queries
    avg_relevance = total_relevance / num_queries
    hallucination_rate = hallucination_count / num_queries
    avg_prompt_quality = total_prompt_quality / num_queries
    
    print("="*97)
    print("                               EVALUATION METRICS SUMMARY")
    print("="*97)
    print(f"Total Predefined Queries Evaluated  : {num_queries}")
    print(f"Average Retrieval Relevance Score   : {avg_relevance:.2%}")
    print(f"Measured Hallucination Rate         : {hallucination_rate:.2%}")
    print(f"Average System Latency              : {avg_latency:.4f} seconds")
    print(f"Average Response Formatting Quality : {avg_prompt_quality:.2%}")
    print("="*97)
    
    return {
        "num_queries": num_queries,
        "average_relevance": avg_relevance,
        "hallucination_rate": hallucination_rate,
        "average_latency": avg_latency,
        "average_prompt_quality": avg_prompt_quality,
        "detailed_results": results
    }
