"""
Text preprocessing utilities for the FAQ chatbot.

Uses NLTK for tokenization, stopword removal, and lemmatization.
Run `python setup_nltk.py` once before first use to download required corpora.
"""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_lemmatizer = WordNetLemmatizer()

try:
    _STOPWORDS = set(stopwords.words("english"))
except LookupError as exc:
    raise RuntimeError(
        "NLTK stopwords corpus not found. Run `python setup_nltk.py` first."
    ) from exc

# NOTE: we deliberately do NOT keep interrogative words ("what", "how", "is",
# etc.) even though they carry some meaning. In testing, keeping them caused
# false-positive matches: two totally unrelated questions that both start
# with "What is...?" would score artificially high on shared bigrams like
# "what is", crowding out the actual content words. Standard stopword
# removal gives more reliable matches on a small FAQ corpus like this one.

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace."""
    text = text.lower().strip()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOPWORDS]


def lemmatize(tokens: list[str]) -> list[str]:
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str) -> str:
    """
    Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize.
    Returns a single space-joined string, ready for vectorization.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    samples = [
        "How do I reset my password?",
        "What's the cost of the Pro plan??",
        "Can I  cancel   my subscription anytime?",
    ]
    for s in samples:
        print(f"{s!r:45} -> {preprocess(s)!r}")
