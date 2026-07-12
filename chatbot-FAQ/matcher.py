"""
FAQ matching engine.

Vectorizes preprocessed FAQ questions with TF-IDF and matches an incoming
user query against them using cosine similarity. Falls back gracefully
when no FAQ is a confident enough match.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess


@dataclass
class MatchResult:
    question: str
    answer: str
    score: float
    is_confident: bool


class FAQMatcher:
    """
    TF-IDF + cosine similarity matcher over a fixed set of FAQs.

    ngram_range=(1,2) lets the vectorizer pick up short two-word phrases
    ("reset password", "free trial") in addition to single words, which
    noticeably improves matches on short questions.
    """

    def __init__(self, faqs: list[dict], confidence_threshold: float = 0.3):
        if not faqs:
            raise ValueError("faqs list cannot be empty")

        self.faqs = faqs
        self.confidence_threshold = confidence_threshold
        self.questions = [f["question"] for f in faqs]
        self.answers = [f["answer"] for f in faqs]

        # Fold optional "tags" (alternate phrasings/synonyms) into the text
        # that gets vectorized, so a question like "Do you have a mobile app?"
        # can still match an FAQ titled "Which devices are supported?" even
        # though the two share almost no vocabulary on their own.
        corpus_texts = []
        for f in faqs:
            text = f["question"] + " " + " ".join(f.get("tags", []))
            corpus_texts.append(text)

        self._processed_questions = [preprocess(t) for t in corpus_texts]

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._faq_matrix = self.vectorizer.fit_transform(self._processed_questions)

    def match(self, user_query: str, top_k: int = 1) -> list[MatchResult]:
        """Return the top_k best-matching FAQs for a user query, ranked by score."""
        processed_query = preprocess(user_query)
        if not processed_query.strip():
            return [MatchResult("", "I didn't catch a question there — could you rephrase?", 0.0, False)]

        query_vec = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vec, self._faq_matrix).flatten()

        ranked_idx = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in ranked_idx:
            score = float(similarities[idx])
            results.append(
                MatchResult(
                    question=self.questions[idx],
                    answer=self.answers[idx],
                    score=score,
                    is_confident=score >= self.confidence_threshold,
                )
            )
        return results

    def best_match(self, user_query: str) -> MatchResult:
        return self.match(user_query, top_k=1)[0]
