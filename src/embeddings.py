"""
TF-IDF embedding generator.
Uses scikit-learn's TfidfVectorizer for lightweight, local embeddings.
"""
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfEmbeddingModel:
    """
    Wraps TfidfVectorizer to provide vector representations for queries and document chunks.
    """
    def __init__(self):
        # Using standard English stop words and sublinear TF scaling for query matching improvements
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            sublinear_tf=True,
            lowercase=True
        )
        self.is_fitted = False

    def fit(self, texts: List[str]) -> None:
        """
        Fits the TfidfVectorizer on the provided list of texts.
        
        Args:
            texts (List[str]): Corpus of text chunks to fit vocabulary.
        """
        if not texts:
            raise ValueError("Cannot fit embedding model on empty text list.")
        self.vectorizer.fit(texts)
        self.is_fitted = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Transforms a list of texts into TF-IDF vector representations.
        
        Args:
            texts (List[str]): Texts to embed.
            
        Returns:
            np.ndarray: Dense 2D numpy array of shape (num_texts, vocabulary_size).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before embedding texts.")
        # Transform to sparse matrix and convert to a dense array for easier calculation
        return self.vectorizer.transform(texts).toarray()

    def embed_query(self, query: str) -> np.ndarray:
        """
        Transforms a search query into a 1D TF-IDF vector.
        
        Args:
            query (str): The search query.
            
        Returns:
            np.ndarray: Dense 1D numpy array of shape (vocabulary_size,).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before embedding a query.")
        # Transform returns shape (1, vocabulary_size), extract the single row
        return self.vectorizer.transform([query]).toarray()[0]
