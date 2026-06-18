from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def evaluate_summaries(original, extractive, abstractive):
    """
    Evaluate summaries by comparing them with the original text
    Returns evaluation metrics
    """
    try:
        # Tokenize into sentences and words
        original_words = set(word_tokenize(original.lower()))
        extractive_words = set(word_tokenize(extractive.lower()))
        abstractive_words = set(word_tokenize(abstractive.lower()))
        
        # Remove stopwords for better comparison
        stop_words = set(stopwords.words('english'))
        original_words = {w for w in original_words if w.isalnum() and w not in stop_words}
        extractive_words = {w for w in extractive_words if w.isalnum() and w not in stop_words}
        abstractive_words = {w for w in abstractive_words if w.isalnum() and w not in stop_words}
        
        # Calculate overlap ratios
        extractive_overlap = len(original_words & extractive_words) / len(original_words) if original_words else 0
        abstractive_overlap = len(original_words & abstractive_words) / len(original_words) if original_words else 0
        
        return {
            "original_length": len(original),
            "extractive_length": len(extractive),
            "abstractive_length": len(abstractive),
            "extractive_word_overlap": round(extractive_overlap * 100, 2),
            "abstractive_word_overlap": round(abstractive_overlap * 100, 2)
        }
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {
            "original_length": len(original),
            "extractive_length": len(extractive),
            "abstractive_length": len(abstractive),
            "error": str(e)
        }