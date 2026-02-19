"""
Text preprocessing utilities for sentiment analysis.
"""

import re
import string
import logging
from typing import List

# Try to import NLTK stopwords, download if necessary
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and preprocess text for sentiment analysis.
    
    Steps:
    1. Convert to lowercase
    2. Remove HTML tags
    3. Remove URLs
    4. Remove special characters and numbers
    5. Remove extra whitespace
    6. Remove stopwords
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in STOPWORDS and len(word) > 2]
    text = ' '.join(words)
    
    return text


def batch_clean_texts(texts: List[str]) -> List[str]:
    """
    Clean a batch of texts.
    
    Args:
        texts: List of raw texts
        
    Returns:
        List of cleaned texts
    """
    return [clean_text(text) for text in texts]

