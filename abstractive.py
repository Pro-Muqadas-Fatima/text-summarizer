"""
Abstractive Text Summarizer (Flask Compatible + Stable)
Uses HuggingFace BART model with lazy loading
"""

from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

# -----------------------------
# NLTK setup
# -----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# -----------------------------
# Lazy model loader (IMPORTANT FIX)
# -----------------------------
_summarizer = None


def get_model():
    global _summarizer

    if _summarizer is None:
        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )

    return _summarizer


# -----------------------------
# MAIN FUNCTION (USED BY FLASK)
# -----------------------------
def abstractive_summarizer(text, num_sentences=3):
    """
    Abstractive summarizer (AI-generated summary)

    NOTE:
    - num_sentences is approximate only
    - BART does NOT strictly control sentence count
    """

    if not text or len(text.split()) < 10:
        return text

    try:
        model = get_model()

        # -----------------------------
        # Handle long text (chunking)
        # -----------------------------
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sent in sentences:
            if len((current_chunk + sent).split()) < 450:
                current_chunk += " " + sent
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sent

        if current_chunk:
            chunks.append(current_chunk.strip())

        # -----------------------------
        # Summarize each chunk
        # -----------------------------
        results = []

        for chunk in chunks:
            summary = model(
                chunk,
                max_length=130,
                min_length=30,
                do_sample=False
            )[0]["summary_text"]

            results.append(summary)

        # -----------------------------
        # Merge summaries
        # -----------------------------
        final_summary = " ".join(results)

        # Optional sentence trimming
        final_sentences = sent_tokenize(final_summary)
        final_sentences = final_sentences[:num_sentences]

        return " ".join(final_sentences)

    except Exception as e:
        print(f"[Abstractive Error] {e}")
        return text