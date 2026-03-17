"""
evaluate.py
Evaluate the TF-IDF + LogReg classifier against the PAN-2012 test set.

Usage:
    python scripts/evaluate.py                   # full test set
    python scripts/evaluate.py --limit 500       # quick smoke-test
    python scripts/evaluate.py --threshold 0.5   # custom decision threshold
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

from lxml import etree

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent.parent
TEST_XML = (
    BASE
    / "pan12-sexual-predator-identification-test-corpus-2012-05-21"
    / "pan12-sexual-predator-identification-test-corpus-2012-05-21"
    / "pan12-sexual-predator-identification-test-corpus-2012-05-17.xml"
)
GROUNDTRUTH_TXT = (
    BASE
    / "pan12-sexual-predator-identification-test-corpus-2012-05-21"
    / "pan12-sexual-predator-identification-test-corpus-2012-05-21"
    / "pan12-sexual-predator-identification-groundtruth-problem1.txt"
)
CLASSIFIER_PATH = BASE / "models" / "classifier" / "tfidf_logreg.pkl"

# ---------------------------------------------------------------------------
# Sliding window config (same as training)
# ---------------------------------------------------------------------------
CONTEXT_WINDOW = 10
STRIDE = 5

_HEX_ID_RE = re.compile(r"\b[0-9a-f]{20,}\b")


def _anonymize(text: str) -> str:
    seen = {}
    counter = [0]

    def _replace(m):
        uid = m.group(0)
        if uid not in seen:
            seen[uid] = f"user_{chr(65 + counter[0])}"
            counter[0] += 1
        return seen[uid]

    return _HEX_ID_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Classifier loading
# ---------------------------------------------------------------------------

def load_classifier(path: Path):
    print(f"[+] Loading classifier from {path} ...")
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    print("[+] Classifier loaded.")
    return pipeline


# ---------------------------------------------------------------------------
# Data loading (reused from original)
# ---------------------------------------------------------------------------

def load_groundtruth(path: Path) -> set[str]:
    predators = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            uid = line.strip()
            if uid:
                predators.add(uid)
    print(f"[+] Loaded {len(predators)} known predators from groundtruth.")
    return predators


def load_test_conversations(xml_path: Path, limit: int | None) -> list[dict]:
    conversations = []
    print(f"[+] Parsing test XML: {xml_path} ...")
    context = etree.iterparse(str(xml_path), events=("end",), tag="conversation")

    for _, conv_elem in context:
        conv_id = conv_elem.get("id", "")
        messages = []
        authors = set()
        for msg_elem in conv_elem.findall(".//message"):
            author = (msg_elem.findtext("author") or "").strip()
            text = " ".join((msg_elem.findtext("text") or "").split())
            if text:
                messages.append({"author": author, "text": text})
                authors.add(author)
        if messages:
            conversations.append({"conv_id": conv_id, "messages": messages, "authors": authors})
        conv_elem.clear()
        if limit and len(conversations) >= limit:
            break

    print(f"[+] Loaded {len(conversations)} test conversations.")
    return conversations


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_conversation(pipeline, messages: list[dict], threshold: float) -> tuple[int, float]:
    """
    Slide windows over the conversation, take max grooming probability.
    Returns (prediction, max_probability).
    """
    max_prob = 0.0

    for start in range(0, len(messages), STRIDE):
        window = messages[start: start + CONTEXT_WINDOW]
        if len(window) < 2:
            continue
        lines = [f"{m['author']}: {m['text']}" for m in window]
        conv_text = _anonymize("\n".join(lines))
        proba = pipeline.predict_proba([conv_text])[0]
        grooming_prob = proba[1]
        if grooming_prob > max_prob:
            max_prob = grooming_prob

    return (1 if max_prob >= threshold else 0), max_prob


# ---------------------------------------------------------------------------
# Metrics (reused from original)
# ---------------------------------------------------------------------------

def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(y_true) if y_true else 0.0

    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def print_results(metrics: dict, total: int):
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Total conversations : {total}")
    print(f"  True Positives      : {metrics['tp']}")
    print(f"  True Negatives      : {metrics['tn']}")
    print(f"  False Positives     : {metrics['fp']}")
    print(f"  False Negatives     : {metrics['fn']}")
    print("-" * 50)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate IMPULSE classifier on PAN-2012 test set.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max conversations to evaluate (for quick smoke-tests).")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for grooming classification (default: 0.5).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not CLASSIFIER_PATH.exists():
        sys.exit("[!] Classifier not found. Run: python scripts/train_classifier.py")

    groundtruth = load_groundtruth(GROUNDTRUTH_TXT)
    conversations = load_test_conversations(TEST_XML, args.limit)
    pipeline = load_classifier(CLASSIFIER_PATH)

    y_true, y_pred = [], []
    total = len(conversations)

    print(f"\n[+] Running inference on {total} conversations (threshold={args.threshold}) ...\n")
    for i, conv in enumerate(conversations, 1):
        is_predator = bool(conv["authors"] & groundtruth)
        y_true.append(1 if is_predator else 0)

        pred, prob = predict_conversation(pipeline, conv["messages"], args.threshold)
        y_pred.append(pred)

        if i % 500 == 0 or i == total:
            done = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            print(f"  [{i}/{total}] running accuracy: {done/i:.3f}")

    metrics = compute_metrics(y_true, y_pred)
    print_results(metrics, total)
