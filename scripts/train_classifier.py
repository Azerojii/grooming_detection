"""
train_classifier.py
Train a TF-IDF + Logistic Regression classifier on the prepared dataset.

Usage:
    python scripts/train_classifier.py

Outputs:
    models/classifier/tfidf_logreg.pkl
"""

import json
import pickle
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models" / "classifier"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches 32-char hex strings (PAN-2012 user IDs)
_HEX_ID_RE = re.compile(r"\b[0-9a-f]{20,}\b")


def extract_sample(record: dict) -> tuple[str, int]:
    """Extract conversation text and label from an SFT JSONL record."""
    messages = record["messages"]
    user_msg = messages[0]["content"]
    assistant_msg = messages[1]["content"]

    # Extract text between <conversation> tags
    match = re.search(r"<conversation>\n(.*?)\n</conversation>", user_msg, re.DOTALL)
    text = match.group(1) if match else user_msg

    # Determine label from assistant response
    label = 0 if "NO GROOMING DETECTED" in assistant_msg else 1

    # Anonymize hex user IDs to prevent memorizing hashes
    seen = {}
    counter = [0]

    def _replace(m):
        uid = m.group(0)
        if uid not in seen:
            seen[uid] = f"user_{chr(65 + counter[0])}"
            counter[0] += 1
        return seen[uid]

    text = _HEX_ID_RE.sub(_replace, text)
    return text, label


def load_dataset(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            text, label = extract_sample(json.loads(line))
            texts.append(text)
            labels.append(label)
    return texts, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[+] Loading training data ...")
    X_train, y_train = load_dataset(DATA_DIR / "train.jsonl")
    print(f"    Train: {len(X_train)} samples ({sum(y_train)} grooming, {len(y_train) - sum(y_train)} normal)")

    print("[+] Loading validation data ...")
    X_val, y_val = load_dataset(DATA_DIR / "val.jsonl")
    print(f"    Val:   {len(X_val)} samples ({sum(y_val)} grooming, {len(y_val) - sum(y_val)} normal)")

    print("[+] Training TF-IDF + Logistic Regression pipeline ...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2), # Reduce noise from rare 3-grams
            sublinear_tf=True,
            min_df=5, # Increase min_df to ignore rare words specific to few convos
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0, # Normal regularization
            class_weight={0: 1.0, 1: 1.5}, # Slight weight boost for grooming without artificially jumping to 78% on normal
            solver="lbfgs",
        )),
    ])

    pipeline.fit(X_train, y_train)
    print("[+] Training complete.")

    print("\n[+] Validation results:")
    y_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=["normal", "grooming"]))

    out_path = MODEL_DIR / "tfidf_logreg.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[+] Saved pipeline to {out_path}")
