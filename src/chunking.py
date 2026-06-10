"""
Document chunking module.
Splits documents into overlapping chunks with metadata.
"""
from typing import List, Dict, Any

class TextChunker:
    """
    Splits documents into overlapping text chunks with clean boundaries.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initializes the TextChunker.
        
        Args:
            chunk_size (int): Max character length per chunk.
            chunk_overlap (int): Overlap character length between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits a single document into a list of chunk dictionaries.
        
        Args:
            doc (Dict[str, Any]): A document dictionary containing:
                - document_id
                - document_name
                - text
                - metadata
                
        Returns:
            List[Dict[str, Any]]: List of chunk dictionaries.
        """
        text = doc["text"]
        doc_id = doc["document_id"]
        doc_name = doc["document_name"]
        
        chunks = []
        if not text or not text.strip():
            return chunks
            
        start = 0
        chunk_index = 0
        text_len = len(text)
        
        while start < text_len:
            # Determine initial end of window
            end = min(start + self.chunk_size, text_len)
            
            # Align end with word boundaries (spaces) to prevent truncating words
            if end < text_len:
                # Look for a space in the last 50 characters of the window
                space_index = text.rfind(' ', max(start, end - 50), end)
                if space_index != -1 and space_index > start:
                    end = space_index
            
            chunk_text = text[start:end].strip()
            
            # Avoid inserting empty chunks
            if chunk_text:
                chunk_id = f"{doc_id}_c{chunk_index}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "chunk_text": chunk_text,
                    "chunk_index": chunk_index,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "start_char": start,
                        "end_char": end,
                        "char_length": len(chunk_text),
                        "word_count": len(chunk_text.split())
                    }
                })
                chunk_index += 1
            
            # Slide window forward. The new start is end minus the overlap
            start = end - self.chunk_overlap
            
            # Safety guards to prevent infinite loops or redundant trailing short chunks
            if start >= text_len - self.chunk_overlap or end >= text_len:
                break
                
        return chunks

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits multiple documents into a flat list of chunk dictionaries.
        
        Args:
            documents (List[Dict[str, Any]]): List of document dictionaries.
            
        Returns:
            List[Dict[str, Any]]: List of all chunk dictionaries.
        """
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks
