"""
Orchestrator runner for the Enterprise GenAI RAG and Agentic Workflow System.
Loads documents, chunks, builds vector embeddings, indexes them, runs queries, and runs evaluation.
"""
import os
import sys
import webbrowser
from flask import Flask, request, jsonify, render_template
from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, SCORE_THRESHOLD
from src.ingestion import DocumentLoader
from src.chunking import TextChunker
from src.embeddings import TfidfEmbeddingModel
from src.vector_store import InMemoryVectorStore
from src.retriever import Retriever
from src.tools import ToolRouter
from src.rag_pipeline import RAGPipeline
from src.agent_workflow import AgentWorkflow
from src.evaluation import run_evaluation

# Initialize Flask App
flask_app = Flask(__name__, template_folder="templates")

# Global pointers to system instances for web queries
web_agent_workflow = None
web_chunks = []
web_documents = []

@flask_app.route("/")
def index():
    return render_template("index.html")

@flask_app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query text provided"}), 400
    res = web_agent_workflow.run(query)
    return jsonify(res)

@flask_app.route("/api/stats", methods=["GET"])
def api_stats():
    from src.tools import metadata_statistics_tool
    stats = metadata_statistics_tool(web_chunks, web_documents)
    return jsonify(stats)

@flask_app.route("/api/documents", methods=["GET"])
def api_documents():
    docs_metadata = []
    for doc in web_documents:
        docs_metadata.append({
            "name": doc["document_name"],
            "size": doc["metadata"].get("file_size", 0)
        })
    return jsonify(docs_metadata)

@flask_app.route("/api/document/<doc_name>", methods=["GET"])
def api_document(doc_name):
    for doc in web_documents:
        if doc["document_name"] == doc_name:
            return doc["text"], 200, {"Content-Type": "text/plain; charset=utf-8"}
    return "Document not found", 404

class Tee:
    """
    Utility class to replicate stdout streams,
    allowing printing to terminal and writing to file simultaneously.
    """
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()

    def flush(self):
        for f in self.files:
            if hasattr(f, 'flush'):
                f.flush()

def main():
    # Setup dual logging to outputs directory
    os.makedirs("outputs", exist_ok=True)
    output_file_path = os.path.join("outputs", "sample_run_output.txt")
    output_file = open(output_file_path, "w", encoding="utf-8")
    
    # Keep the original stdout to restore it later if needed
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, output_file)
    
    try:
        print("="*80)
        print("          ENTERPRISE GENAI RAG & AGENTIC WORKFLOW SYSTEM INITIALIZATION")
        print("="*80)
        
        # 1. Ingestion: Load raw text files
        print("[1/5] Ingesting documents from data/sample_documents/...")
        doc_dir = os.path.join("data", "sample_documents")
        loader = DocumentLoader(doc_dir)
        documents = loader.load_documents()
        print(f"      Successfully loaded {len(documents)} document files.")
        for d in documents:
            print(f"      - {d['document_name']} ({len(d['text'])} characters)")
            
        # 2. Chunking: Split documents into overlapping chunks
        print("\n[2/5] Chunking documents (Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP})...")
        chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = chunker.split_documents(documents)
        print(f"      Generated {len(chunks)} overlapping text chunks.")
        
        # 3. Embeddings: Fit and build TF-IDF representations
        print("\n[3/5] Building TF-IDF Vector Embeddings...")
        embedding_model = TfidfEmbeddingModel()
        chunk_texts = [c["chunk_text"] for c in chunks]
        embedding_model.fit(chunk_texts)
        chunk_vectors = embedding_model.embed_texts(chunk_texts)
        print(f"      Vocabulary size: {len(embedding_model.vectorizer.get_feature_names_out())} words.")
        
        # 4. Indexing: Populate the in-memory vector store
        print("\n[4/5] Indexing chunks in local InMemoryVectorStore...")
        vector_store = InMemoryVectorStore()
        vector_store.add_chunks(chunks, chunk_vectors)
        print("      Index populated successfully.")
        
        # 5. Pipeline Setup: Retriever, Tools, RAG, and Workflow
        print("\n[5/5] Initializing agent workflow components...")
        retriever = Retriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            score_threshold=SCORE_THRESHOLD
        )
        tool_router = ToolRouter(
            retriever=retriever,
            chunks=chunks,
            documents=documents
        )
        rag_pipeline = RAGPipeline()
        agent_workflow = AgentWorkflow(
            retriever=retriever,
            tool_router=tool_router,
            rag_pipeline=rag_pipeline
        )
        print("      Agent workflow components ready.")
        print("="*80)
        
        # Run Demo queries
        print("\n" + "="*80)
        print("                      RUNNING SAMPLE WORKFLOW DEMO QUERIES")
        print("="*80)
        
        demo_queries = [
            "How can an employee request remote work?",
            "What should I do if I cannot access VPN?",
            "What are the sales pipeline stages?",
            "Show metadata statistics.",
            "What support options are available for customers?"
        ]
        
        for idx, query in enumerate(demo_queries, 1):
            print(f"\n[DEMO QUERY {idx}]: \"{query}\"")
            result = agent_workflow.run(query)
            
            # Print execution details
            print(f"  └─ Stage 1 (Handling): Cleaned Query, classified type as '{result['query_type']}'")
            print(f"  └─ Stage 2 (Retrieval): Retracted {len(result['retrieved_context'])} relevant chunks above threshold")
            print(f"  └─ Stage 3 (Routing): Routed to '{result['selected_tool']}'")
            print(f"  └─ Stage 4 (Response): Generated with '{result['confidence']}' confidence in {result['latency_seconds']:.4f}s")
            
            # Print retrieved document citations
            if result['retrieved_context']:
                sources = {c['document_name'] for c in result['retrieved_context']}
                print(f"  └─ Cited Documents: {', '.join(sources)}")
                
            print("\n  [Final Generated Answer]:")
            # Indent answer for nice display formatting
            indented_answer = "\n".join("      " + line for line in result["answer"].split("\n"))
            print(indented_answer)
            print("-" * 80)
            
        # Run System Evaluation Summary
        run_evaluation(agent_workflow)
        
        print(f"\n[Success] Execution complete. Log output written to: {output_file_path}")
        
        # Assign values for the global web service pointers
        global web_agent_workflow, web_chunks, web_documents
        web_agent_workflow = agent_workflow
        web_chunks = chunks
        web_documents = documents
        
    finally:
        # Restore normal stdout
        sys.stdout = original_stdout
        output_file.close()

    # Run Flask application for user interaction
    print("\n" + "="*80)
    print("                STARTING INTERACTIVE WEB CHAT INTERFACE")
    print("="*80)
    print("  * Hosting local web page at: http://127.0.0.1:8000")
    print("  * Automatically launching default browser view...")
    
    # Auto open browser
    webbrowser.open("http://127.0.0.1:8000")
    
    # Start web server
    flask_app.run(host="127.0.0.1", port=8000, debug=False)

if __name__ == "__main__":
    main()
