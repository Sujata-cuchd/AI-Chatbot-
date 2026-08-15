"""
AI Chatbot using NLP (text preprocessing) + Machine Learning (intent classification).

How it works
------------
1. Training examples ("patterns") for each intent are loaded from intents.json.
2. Each pattern is cleaned/normalized (lowercased, punctuation stripped, light
   stemming) — this is the NLP preprocessing step.
3. Patterns are converted into TF-IDF vectors and used to train a classifier
   (Logistic Regression) that predicts which "intent" a new message belongs to
   — this is the ML step.
4. At chat time, the user's message is classified into an intent, and a
   response is picked (randomly) from that intent's response list.
5. If the model isn't confident enough, a fallback response is used instead
   of guessing wildly.

No external NLP downloads (like nltk's punkt) are required — preprocessing is
done with plain regex + a small custom stemmer, so this runs out of the box
with just scikit-learn installed.

Usage
-----
    python chatbot.py            # train (if needed) and start chatting
    python chatbot.py --retrain  # force retraining even if a saved model exists
"""

import json
import re
import random
import pickle
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = BASE_DIR / "intents.json"
MODEL_PATH = BASE_DIR / "chatbot_model.pkl"

CONFIDENCE_THRESHOLD = 0.22  # below this, we fall back instead of guessing


# ---------------------------------------------------------------------------
# NLP preprocessing
# ---------------------------------------------------------------------------

# A tiny hand-rolled suffix stripper. It's not as thorough as a real stemmer
# (e.g. NLTK's PorterStemmer) but needs no extra downloads and works fine for
# a small-vocabulary chatbot.
_SUFFIXES = ("ing", "edly", "edness", "ed", "ly", "es", "s")


def simple_stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def preprocess(text: str) -> str:
    """Lowercase, strip punctuation/digits, tokenize, and lightly stem."""
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    tokens = text.split()
    stemmed = [simple_stem(tok) for tok in tokens]
    return " ".join(stemmed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_intents():
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["intents"]


def build_training_data(intents):
    texts, labels = [], []
    for intent in intents:
        for pattern in intent["patterns"]:
            texts.append(preprocess(pattern))
            labels.append(intent["tag"])
    return texts, labels


# ---------------------------------------------------------------------------
# Model training / loading
# ---------------------------------------------------------------------------

def train_model(intents):
    texts, labels = build_training_data(intents)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(texts, labels)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Model trained on {len(texts)} examples across {len(set(labels))} intents.")
    return pipeline


def load_or_train_model(intents, force_retrain=False):
    if MODEL_PATH.exists() and not force_retrain:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_model(intents)


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

def get_response(model, intents, user_message: str) -> str:
    cleaned = preprocess(user_message)

    if not cleaned.strip():
        return "Could you say that again?"

    probs = model.predict_proba([cleaned])[0]
    classes = model.classes_
    best_idx = probs.argmax()
    best_tag = classes[best_idx]
    confidence = probs[best_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        best_tag = "fallback"

    for intent in intents:
        if intent["tag"] == best_tag:
            return random.choice(intent["responses"])

    return "I'm not sure how to respond to that."


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def chat():
    intents = load_intents()
    force_retrain = "--retrain" in sys.argv
    model = load_or_train_model(intents, force_retrain=force_retrain)

    print("\nChatbot is ready! Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        if user_message.lower() in {"quit", "exit"}:
            print("Bot: Goodbye!")
            break

        response = get_response(model, intents, user_message)
        print(f"Bot: {response}")


if __name__ == "__main__":
    chat()
