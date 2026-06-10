"""
Document ingestion module.
Loads documents from a target directory.
"""
import hashlib
from typing import List, Dict, Any
from pathlib import Path

class DocumentLoader:
    """
    Loads text files from a directory and extracts content and metadata.
    """
    def __init__(self, directory_path: str):
        self.directory_path = Path(directory_path)

    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Loads all .txt files in the specified directory.
        
        Returns:
            List[Dict[str, Any]]: List of document dictionaries.
            Each dictionary contains:
                - document_id: Unique hash identifier
                - document_name: Base filename
                - text: Unstructured raw document text
                - metadata: Dictionary of document properties (file path, size)
            
        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        if not self.directory_path.exists():
            raise FileNotFoundError(
                f"Document directory '{self.directory_path}' does not exist. "
                "Please verify the path and ensure data/sample_documents is populated."
            )
            
        documents = []
        # Support globbing for text files
        for file_path in sorted(self.directory_path.glob("*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                doc_name = file_path.name
                # Generate a deterministic unique ID using MD5 hash of filename
                doc_id = hashlib.md5(doc_name.encode("utf-8")).hexdigest()[:8]
                
                documents.append({
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "text": text,
                    "metadata": {
                        "file_path": str(file_path),
                        "file_size": file_path.stat().st_size,
                    }
                })
            except Exception as e:
                print(f"Error loading document {file_path}: {e}")
                
        return documents
