"""
prepare_dataset.py
Parse PAN-2012 training XML and generate SFT JSONL files for fine-tuning.

Usage:
    python scripts/prepare_dataset.py

Outputs:
    data/train.jsonl
    data/val.jsonl
"""

import json
import random
import re
from pathlib import Path

from lxml import etree

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent.parent
TRAIN_XML = (
    BASE
    / "pan12-sexual-predator-identification-training-corpus-2012-05-01"
    / "pan12-sexual-predator-identification-training-corpus-2012-05-01"
    / "pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
)
PREDATORS_TXT = (
    BASE
    / "pan12-sexual-predator-identification-training-corpus-2012-05-01"
    / "pan12-sexual-predator-identification-training-corpus-2012-05-01"
    / "pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"
)
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONTEXT_WINDOW = 10          # number of recent messages to include as context
VAL_FRACTION = 0.15
RANDOM_SEED = 42
MAX_NORMAL_SAMPLES = None    # set to int to cap normal samples (None = auto-balance)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_predator_ids(path: Path) -> set:
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            uid = line.strip()
            if uid:
                ids.add(uid)
    print(f"[+] Loaded {len(ids)} predator IDs")
    return ids


def clean_text(text: str) -> str:
    """Normalise whitespace and remove control characters."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return " ".join(text.split())


def messages_to_context(messages: list[dict]) -> str:
    """Format a list of {author, text} dicts into a readable block."""
    lines = []
    for m in messages:
        author = m["author"] or "unknown"
        text = m["text"] or ""
        lines.append(f"{author}: {text}")
    return "\n".join(lines)


def make_sft_sample(context_text: str, is_grooming: bool) -> dict:
    """Wrap context + label into the chat SFT format."""
    user_prompt = (
        "Analyze the following conversation excerpt for signs of child grooming "
        "or sexual predatory behavior. Consider manipulation tactics, boundary "
        "violations, age-inappropriate content, and trust-building patterns.\n\n"
        "<conversation>\n"
        f"{context_text}\n"
        "</conversation>"
    )

    if is_grooming:
        assistant_response = (
            "GROOMING DETECTED: This conversation exhibits predatory behavior patterns. "
            "Indicators include: attempts to isolate the target, age-inappropriate "
            "discussions, trust-building manipulation, boundary testing, and/or "
            "requests for personal information or meetings. The language used suggests "
            "intentional grooming of a minor. Risk level: HIGH."
        )
    else:
        assistant_response = (
            "NO GROOMING DETECTED: This conversation does not show clear indicators "
            "of predatory or grooming behavior. The exchanges appear to be typical "
            "social interaction without manipulation tactics, boundary violations, or "
            "inappropriate content targeting a minor. Risk level: LOW."
        )

    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


# ---------------------------------------------------------------------------
# Main parsing logic
# ---------------------------------------------------------------------------

def parse_conversations(xml_path: Path, predator_ids: set) -> tuple[list, list]:
    """
    Returns two lists: grooming_samples, normal_samples.
    Each sample is a dict ready to be written as a JSONL line.
    """
    grooming_samples = []
    normal_samples = []

    print(f"[+] Parsing {xml_path} ...")
    context = etree.iterparse(str(xml_path), events=("end",), tag="conversation")

    conv_count = 0
    for _, conv_elem in context:
        conv_id = conv_elem.get("id", "")
        messages = []

        for msg_elem in conv_elem.findall(".//message"):
            author = (msg_elem.findtext("author") or "").strip()
            text = clean_text(msg_elem.findtext("text") or "")
            if text:
                messages.append({"author": author, "text": text})

        if not messages:
            conv_elem.clear()
            continue

        # Determine if this conversation involves a known predator
        authors_in_conv = {m["author"] for m in messages}
        is_grooming = bool(authors_in_conv & predator_ids)

        # Slide a context window over the conversation
        step = max(1, CONTEXT_WINDOW // 2)
        for start in range(0, len(messages), step):
            window = messages[start : start + CONTEXT_WINDOW]
            if len(window) < 2:
                continue
            context_text = messages_to_context(window)
            sample = make_sft_sample(context_text, is_grooming)
            if is_grooming:
                grooming_samples.append(sample)
            else:
                normal_samples.append(sample)

        conv_elem.clear()
        conv_count += 1
        if conv_count % 5000 == 0:
            print(f"    ... processed {conv_count} conversations")

    print(f"[+] Processed {conv_count} conversations total")
    print(f"    Grooming samples : {len(grooming_samples)}")
    print(f"    Normal samples   : {len(normal_samples)}")
    return grooming_samples, normal_samples


def balance_and_split(
    grooming: list, normal: list, val_fraction: float, seed: int
) -> tuple[list, list]:
    rng = random.Random(seed)

    # Balance: 1:1 ratio to prevent model collapsing to majority class
    target_normal = min(len(normal), len(grooming))
    normal_balanced = rng.sample(normal, target_normal)

    all_samples = grooming + normal_balanced
    rng.shuffle(all_samples)

    split = int(len(all_samples) * (1 - val_fraction))
    train = all_samples[:split]
    val = all_samples[split:]

    print(f"[+] Final split — train: {len(train)}, val: {len(val)}")
    return train, val


def write_jsonl(samples: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[+] Wrote {len(samples)} samples -> {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    predator_ids = load_predator_ids(PREDATORS_TXT)
    grooming_samples, normal_samples = parse_conversations(TRAIN_XML, predator_ids)
    train, val = balance_and_split(grooming_samples, normal_samples, VAL_FRACTION, RANDOM_SEED)
    write_jsonl(train, DATA_DIR / "train.jsonl")
    write_jsonl(val, DATA_DIR / "val.jsonl")
    print("[+] Dataset preparation complete.")
