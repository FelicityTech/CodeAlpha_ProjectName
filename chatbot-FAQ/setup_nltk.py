"""Run this once to download the NLTK corpora the chatbot needs."""
import nltk

PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]

if __name__ == "__main__":
    for pkg in PACKAGES:
        nltk.download(pkg)
    print("NLTK setup complete.")
