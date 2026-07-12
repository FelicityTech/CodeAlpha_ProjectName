# FAQ Chatbot — NLP-Based Question Matching

A self-contained FAQ chatbot that answers user questions by matching them
against a curated set of FAQs using classic NLP techniques — NLTK for text
preprocessing and TF-IDF + cosine similarity for matching. Includes both a
terminal interface and a Gradio-based web chat UI.

Sample data is a fictional cloud storage product ("CloudSync") with 25 Q&A
pairs, but the whole pipeline is topic-agnostic — swap in your own FAQs and
it works the same way.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Setup](#setup)
- [Usage](#usage)
- [Customizing for Your Own FAQs](#customizing-for-your-own-faqs)
- [Testing](#testing)
- [Tuning the Confidence Threshold](#tuning-the-confidence-threshold)
- [Known Limitations](#known-limitations)
- [Possible Improvements](#possible-improvements)
- [Tech Stack](#tech-stack)

---

## Features

- **NLP preprocessing** — lowercasing, punctuation stripping, tokenization,
  stopword removal, and lemmatization via NLTK
- **Similarity-based matching** — TF-IDF vectorization (unigrams + bigrams)
  with cosine similarity, not simple keyword search
- **Tag-based recall boosting** — FAQs can carry alternate phrasings/synonyms
  that widen what a question can match to, without changing the canonical
  question or answer shown to the user
- **Confidence thresholding** — the bot won't confidently return a wrong
  answer; below the threshold it admits uncertainty and offers its best
  guesses instead
- **Two interfaces** — a terminal chat loop (`chatbot.py`) and a Gradio web
  UI (`app.py`)
- **A real test suite** — 14 cases covering exact matches, paraphrases,
  off-topic questions, and edge cases, all currently passing

## Project Structure

```
faq_chatbot/
├── data/
│   └── faqs.json         # Q&A pairs (+ optional "tags" per entry)
├── preprocess.py          # NLTK-based text cleaning pipeline
├── matcher.py              # TF-IDF + cosine similarity matching engine
├── chatbot.py               # Orchestration: matching + confidence fallback
├── app.py                    # Gradio web chat UI
├── setup_nltk.py               # One-time NLTK corpus downloader
├── test_chatbot.py               # Verification test suite (run this!)
├── requirements.txt
└── README.md
```

## How It Works

```
data/faqs.json  →  preprocess.py  →  matcher.py  →  chatbot.py  →  app.py
 (Q&A + tags)      (clean, tokenize,   (TF-IDF +    (confidence     (Gradio
                    remove stopwords,   cosine        threshold,     chat UI)
                    lemmatize)          similarity)   fallback msg)
```

### 1. Preprocessing (`preprocess.py`)

Each question — both the FAQ questions and the incoming user query — goes
through the same pipeline:

1. Lowercase and strip punctuation
2. Tokenize (NLTK `word_tokenize`)
3. Remove English stopwords (NLTK's stopword list)
4. Lemmatize each token (NLTK `WordNetLemmatizer`) — e.g. "cancelled" → "cancel"

Interrogative words like "what," "how," and "is" are removed along with
every other stopword. Earlier testing showed that *keeping* them caused
false-positive matches: two completely unrelated questions that both
happened to start with "What is...?" would score artificially high on the
shared sentence stem alone, drowning out the actual content words that
should have driven the match.

### 2. Matching (`matcher.py`)

The cleaned FAQ questions are vectorized with `TfidfVectorizer` (unigrams +
bigrams), and an incoming query is compared against every FAQ using cosine
similarity. The FAQ with the highest score wins.

Each FAQ entry in `faqs.json` can optionally include a `"tags"` list —
alternate phrasings a real user might type that don't share vocabulary with
the canonical question. For example:

```json
{
  "question": "Which devices are supported?",
  "answer": "CloudSync works on Windows, macOS, Linux, iOS, and Android...",
  "tags": ["mobile app", "iphone", "android app", "which platforms"]
}
```

Tags get folded into the text that's vectorized (but never shown to the
user), which meaningfully improves recall on paraphrases — this alone was
the difference between "Do you have a mobile app?" matching correctly vs.
falling back to "no answer found."

### 3. Confidence Handling (`chatbot.py`)

A raw similarity score isn't meaningful on its own — it needs a cutoff.
`FAQChatbot` applies a confidence threshold (default `0.3`, tuned against
the test suite): scores above it return the matched answer directly; scores
below it return a fallback message plus up to three "did you mean...?"
suggestions pulled from the next-best matches.

### 4. Interfaces (`app.py`, `chatbot.py`)

`chatbot.py` can be run directly as a terminal chat loop. `app.py` wraps the
same `FAQChatbot` class in a Gradio `ChatInterface` for a proper web chat
window with example questions and message history.

---

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python setup_nltk.py       # one-time download of NLTK corpora (punkt, stopwords, wordnet)
```

## Usage

**Terminal chat:**
```bash
python chatbot.py
```
Type your question, `quit` or `exit` to stop.

**Web chat UI:**
```bash
python app.py
```
Opens a local Gradio server (prints a URL, usually `http://127.0.0.1:7860`).

**Programmatic use:**
```python
from chatbot import FAQChatbot

bot = FAQChatbot()
print(bot.respond("How do I reset my password?"))

# Or get the raw match details (score, matched question, confidence):
result = bot.respond_verbose("How do I reset my password?")
# {'user_message': '...', 'matched_question': '...', 'answer': '...', 'score': 1.0, 'is_confident': True}
```

## Customizing for Your Own FAQs

1. Edit `data/faqs.json`. Each entry needs `"question"` and `"answer"`;
   `"tags"` is optional but recommended for anything users might phrase
   differently than the canonical question.
2. Re-run `python test_chatbot.py` — you'll want to update the test cases in
   `TEST_CASES` to match your new topic, then use failures the same way this
   project did: as a signal to add tags or adjust the threshold, not
   something to ignore.
3. No code changes needed in `preprocess.py`, `matcher.py`, or `chatbot.py`
   — they're all topic-agnostic.

## Testing

```bash
python test_chatbot.py
```

Runs 14 cases: exact-match questions, realistic paraphrases (different
wording, same intent), clearly off-topic questions (should fall back), and
edge cases (empty input, gibberish). Prints a pass/fail table with the
actual similarity score for each case — useful for seeing *why* a match
succeeded or failed, not just whether it did.

## Tuning the Confidence Threshold

The default threshold (`0.3`) was arrived at empirically, not guessed —
it's the value that made the test suite pass without letting through
false-positive matches on off-topic questions. If you add your own FAQs:

- **Threshold too high** → correct paraphrases get rejected as "not confident"
- **Threshold too low** → off-topic or unrelated questions start getting
  confidently (and wrongly) answered

Adjust it via the `confidence_threshold` parameter on `FAQChatbot()` or
`FAQMatcher()`, and re-run the test suite after any change.

## Known Limitations

TF-IDF and cosine similarity match on shared **vocabulary**, not **meaning**.
The system has no built-in notion that "iphone" and "ios" are related, or
that "get my money back" means the same thing as "refund" — unless that
relationship is captured via a `tags` entry. This works well for a bounded,
well-anticipated FAQ set, but has two concrete failure modes observed during
testing:

1. **Vocabulary gaps** — a paraphrase using words nowhere in the corpus (and
   not covered by tags) won't match, even if the intent is obvious to a
   human reader.
2. **Ambiguous shared words** — a word that's a strong signal for one FAQ
   (e.g. "support" for the customer-support FAQ) can pull in queries that
   are actually about something else entirely (e.g. "which *devices* are
   supported").

These are addressed by adding `tags`, not by trying to make the underlying
similarity metric smarter — see below for that.

## Possible Improvements

- **Semantic embeddings** — swap the `TfidfVectorizer` for sentence
  embeddings (e.g. `sentence-transformers`) and compare embedding vectors
  instead of TF-IDF vectors. Same architecture, genuinely understands
  synonyms and paraphrasing, at the cost of a heavier dependency and slower
  first-load.
- **Multi-turn context** — currently every message is matched independently;
  there's no memory of what was asked previously in the conversation.
  Gradio's `ChatInterface` already receives `history`, it's just unused.
- **Analytics** — log unmatched/low-confidence queries to see what real
  users ask that isn't covered yet, and use that to expand `faqs.json`.
- **Multi-language FAQs** — NLTK's stopword lists and lemmatizer support
  several languages; the pipeline could be extended to detect the query
  language and preprocess accordingly.

## Tech Stack

| Component | Library |
|---|---|
| Tokenization, stopwords, lemmatization | NLTK |
| Vectorization + similarity | scikit-learn (`TfidfVectorizer`, `cosine_similarity`) |
| Web chat UI | Gradio |
| Numerical operations | NumPy |