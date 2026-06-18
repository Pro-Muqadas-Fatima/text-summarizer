from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk

# Download required data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def extractive_summarizer(text, top_n=3):
    """
    Extractive summarizer using frequency-based scoring
    """

    if not text or len(text.split()) < 5:
        return text

    try:
        sentences = sent_tokenize(text)

        if len(sentences) <= top_n:
            return text

        # Tokenize words
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())

        words = [
            w for w in words
            if w.isalnum() and w not in stop_words
        ]

        freq = Counter(words)

        # Sentence scoring
        sentence_scores = {}

        for i, sentence in enumerate(sentences):
            sentence_words = word_tokenize(sentence.lower())

            score = 0
            word_count = 0

            for word in sentence_words:
                if word in freq:
                    score += freq[word]
                    word_count += 1

            # normalize by sentence length (IMPORTANT FIX)
            if word_count > 0:
                sentence_scores[i] = score / word_count

        # Top sentences
        top_indices = sorted(
            sentence_scores,
            key=sentence_scores.get,
            reverse=True
        )[:top_n]

        # Maintain original order
        top_indices = sorted(top_indices)

        summary = ' '.join([sentences[i] for i in top_indices])

        return summary

    except Exception as e:
        print(f"Extractive summarization error: {e}")
        return text