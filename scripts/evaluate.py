"""
evaluate.py
Evaluate the fine-tuned model against the PAN-2012 test set.

Usage:
    python scripts/evaluate.py                   # auto-detect latest checkpoint
    python scripts/evaluate.py --checkpoint 3000 # specific checkpoint
    python scripts/evaluate.py --limit 200       # cap conversations (faster smoke-test)

Outputs:
    Precision, Recall, F1, Accuracy at the conversation level.
    Prints a confusion matrix and per-class breakdown.
"""

import argparse
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
LORA_ADAPTER_DIR = BASE / "models" / "lora_adapter"
MERGED_DIR = BASE / "models" / "merged"
FALLBACK_BASE_MODEL = "Qwen/Qwen3.5-0.8B"

# ---------------------------------------------------------------------------
# Same sliding-window config as training
# ---------------------------------------------------------------------------
CONTEXT_WINDOW = 10
STRIDE = 5

SYSTEM_PROMPT = (
    "You are a child safety AI assistant trained to detect grooming behavior "
    "in online conversations. Analyze conversations carefully and identify "
    "manipulation tactics, boundary violations, age-inappropriate content, "
    "trust-building patterns, isolation attempts, and any other indicators of "
    "predatory behavior targeting minors."
)

ANALYSIS_PROMPT = (
    "Analyze the following conversation excerpt for signs of child grooming "
    "or sexual predatory behavior. Consider manipulation tactics, boundary "
    "violations, age-inappropriate content, and trust-building patterns.\n\n"
    "<conversation>\n{conversation}\n</conversation>\n\n"
    "Respond with exactly one line:\n"
    "VERDICT: HIGH RISK or VERDICT: LOW RISK"
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def find_latest_checkpoint(adapter_dir: Path) -> Path | None:
    checkpoints = sorted(
        [d for d in adapter_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    return checkpoints[-1] if checkpoints else None


def load_model(checkpoint_num: int | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # Determine model path
    if MERGED_DIR.exists() and (MERGED_DIR / "config.json").exists():
        print(f"[+] Using merged model: {MERGED_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(str(MERGED_DIR))
        model = AutoModelForCausalLM.from_pretrained(
            str(MERGED_DIR),
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
    else:
        if checkpoint_num is not None:
            ckpt_path = LORA_ADAPTER_DIR / f"checkpoint-{checkpoint_num}"
            if not ckpt_path.exists():
                sys.exit(f"[!] Checkpoint not found: {ckpt_path}")
        else:
            ckpt_path = find_latest_checkpoint(LORA_ADAPTER_DIR)
            if ckpt_path is None:
                print(f"[!] No checkpoints found. Using base model: {FALLBACK_BASE_MODEL}")
                ckpt_path = None

        if ckpt_path is not None:
            print(f"[+] Using LoRA adapter: {ckpt_path}")
            tokenizer = AutoTokenizer.from_pretrained(FALLBACK_BASE_MODEL)
            base = AutoModelForCausalLM.from_pretrained(
                FALLBACK_BASE_MODEL,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(base, str(ckpt_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(FALLBACK_BASE_MODEL)
            model = AutoModelForCausalLM.from_pretrained(
                FALLBACK_BASE_MODEL,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto",
            )

    model.eval()
    print(f"[+] Model ready on {device}.")
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_window(model, tokenizer, device, conversation_text: str) -> str:
    """Returns 'HIGH' or 'LOW' for a single window."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ANALYSIS_PROMPT.format(conversation=conversation_text)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).upper()
    return "HIGH" if "HIGH" in response else "LOW"


def predict_conversation(model, tokenizer, device, messages: list[dict]) -> str:
    """
    Slide a window over the conversation, aggregate window votes.
    Returns 'HIGH' if any window votes HIGH (conservative for safety).
    """
    high_votes = 0
    total_windows = 0

    for start in range(0, len(messages), STRIDE):
        window = messages[start: start + CONTEXT_WINDOW]
        if len(window) < 2:
            continue
        lines = [f"{m['author']}: {m['text']}" for m in window]
        conv_text = "\n".join(lines)
        verdict = predict_window(model, tokenizer, device, conv_text)
        total_windows += 1
        if verdict == "HIGH":
            high_votes += 1

    # Flag as grooming if >30% of windows vote HIGH
    if total_windows == 0:
        return "LOW"
    return "HIGH" if (high_votes / total_windows) >= 0.3 else "LOW"


# ---------------------------------------------------------------------------
# Data loading
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
    """Returns list of {conv_id, messages, authors}."""
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
# Metrics
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
    parser = argparse.ArgumentParser(description="Evaluate IMPULSE model on PAN-2012 test set.")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Checkpoint number to use (e.g. 3200). Defaults to latest.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max conversations to evaluate (for quick smoke-tests).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    groundtruth = load_groundtruth(GROUNDTRUTH_TXT)
    conversations = load_test_conversations(TEST_XML, args.limit)
    model, tokenizer, device = load_model(args.checkpoint)

    y_true, y_pred = [], []
    total = len(conversations)

    print(f"\n[+] Running inference on {total} conversations ...\n")
    for i, conv in enumerate(conversations, 1):
        # Ground truth: any author in this conversation is a known predator
        is_predator = bool(conv["authors"] & groundtruth)
        y_true.append(1 if is_predator else 0)

        prediction = predict_conversation(model, tokenizer, device, conv["messages"])
        y_pred.append(1 if prediction == "HIGH" else 0)

        if i % 50 == 0 or i == total:
            done = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            print(f"  [{i}/{total}] running accuracy: {done/i:.3f}")

    metrics = compute_metrics(y_true, y_pred)
    print_results(metrics, total)
