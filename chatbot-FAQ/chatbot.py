"""
FAQ Chatbot core logic.

Loads a set of FAQs, builds a matcher, and answers user questions —
falling back to a friendly "I'm not sure" response (with suggestions)
when nothing matches confidently enough.
"""
import json
from pathlib import Path

from matcher import FAQMatcher

DEFAULT_FAQ_PATH = Path(__file__).parent / "data" / "faqs.json"

FALLBACK_MESSAGE = (
    "I'm not sure I have an answer for that. Could you try rephrasing, "
    "or ask about billing, account settings, syncing, or security?"
)


def load_faqs(path: Path = DEFAULT_FAQ_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class FAQChatbot:
    def __init__(self, faq_path: Path = DEFAULT_FAQ_PATH, confidence_threshold: float = 0.3):
        self.faqs = load_faqs(faq_path)
        self.matcher = FAQMatcher(self.faqs, confidence_threshold=confidence_threshold)

    def respond(self, user_message: str, show_suggestions: bool = True) -> str:
        if not user_message or not user_message.strip():
            return "Ask me anything about your account, billing, or how CloudSync works!"

        top_matches = self.matcher.match(user_message, top_k=3)
        best = top_matches[0]

        if best.is_confident:
            return best.answer

        if show_suggestions:
            suggestions = [m.question for m in top_matches if m.score > 0.08]
            if suggestions:
                bullet_list = "\n".join(f"- {q}" for q in suggestions)
                return (
                    f"{FALLBACK_MESSAGE}\n\nDid you mean one of these?\n{bullet_list}"
                )

        return FALLBACK_MESSAGE

    def respond_verbose(self, user_message: str) -> dict:
        """Same as respond(), but also returns the match score — useful for debugging/tests."""
        best = self.matcher.best_match(user_message)
        return {
            "user_message": user_message,
            "matched_question": best.question,
            "answer": best.answer if best.is_confident else FALLBACK_MESSAGE,
            "score": round(best.score, 3),
            "is_confident": best.is_confident,
        }


if __name__ == "__main__":
    bot = FAQChatbot()
    print("FAQ Chatbot ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        print(f"Bot: {bot.respond(user_input)}\n")
