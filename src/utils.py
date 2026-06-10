"""
Utility functions for text processing.
"""
import re

def clean_text(text: str) -> str:
    """
    Cleans text by converting to lowercase, removing extra whitespaces,
    and removing newlines.
    
    Args:
        text (str): Input text to clean.
        
    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text
