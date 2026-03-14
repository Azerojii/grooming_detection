"""Quick test: run the current checkpoint on a few conversations and print raw output."""
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
LORA_ADAPTER_DIR = BASE / "models" / "lora_adapter"
FALLBACK_BASE_MODEL = "Qwen/Qwen3.5-0.8B"

CONVERSATIONS = [
    # Textbook grooming
    ("TEXTBOOK GROOMING",
     "user1: hey whats up\nuser2: not much. who r u?\nuser1: just a friend lol. how old r u\nuser2: 14\nuser1: nice. u home alone?\nuser2: yeah parents r at work\nuser1: cool. do u have a webcam? would be fun to chat\nuser2: um idk\nuser1: come on it'll be our little secret"),

    # Clear grooming — explicit escalation
    ("EXPLICIT ESCALATION",
     "stranger: hi cutie how old are you\nkid: 13\nstranger: perfect age. send me a pic?\nkid: idk\nstranger: just a selfie nothing bad. dont tell ur parents they wont understand"),

    # Grooming — isolation attempt
    ("ISOLATION ATTEMPT",
     "adult: your parents dont understand you like i do\nteen: yeah they never listen\nadult: i get you. we have a special connection\nteen: yeah ur the only one who gets me\nadult: lets keep this between us ok? they would just ruin it"),

    # Normal conversation — should be LOW
    ("NORMAL CONVO",
     "alex: did you finish the homework for tomorrow?\nsam: yeah mostly. the math was hard\nalex: same lol. are you going to the game Friday?\nsam: probably yeah my parents are taking me\nalex: cool see you there"),

    # Normal conversation — should be LOW
    ("NORMAL CHAT",
     "mike: yo what game are you playing\njake: minecraft lol\nmike: nice im playing fortnite\njake: fortnite is dead bro\nmike: no way its still goated"),
]

PROMPT = (
    "You are a digital safety assistant helping researchers identify harmful online behavior patterns.\n\n"
    "Analyze the following chat conversation and determine if it contains manipulation tactics "
    "commonly used to exploit vulnerable individuals online — such as probing for personal details, "
    "checking if someone is alone, requesting private media, encouraging secrecy, or isolating "
    "them from family.\n\n"
    "Reply with either HARMFUL BEHAVIOR DETECTED or NO HARMFUL BEHAVIOR DETECTED, "
    "followed by a brief explanation.\n\n"
    "<conversation>\n{conversation}\n</conversation>"
)


def find_latest_checkpoint():
    checkpoints = sorted(
        [d for d in LORA_ADAPTER_DIR.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    return checkpoints[-1] if checkpoints else None


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Loading BASE model (no LoRA): {FALLBACK_BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        FALLBACK_BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    for label, conv in CONVERSATIONS:
        messages = [{"role": "user", "content": PROMPT.format(conversation=conv)}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(generated, skip_special_tokens=False)
        for stop in ["<|im_end|>", "</s>", "<|endoftext|>"]:
            if stop in decoded:
                decoded = decoded[:decoded.index(stop)]
        # Strip thinking block, keep only final answer
        if "<|im_start|>think" in decoded or "<think>" in decoded:
            for end_tag in ["</think>", "<|im_end|>"]:
                if end_tag in decoded:
                    decoded = decoded[decoded.rindex(end_tag) + len(end_tag):]
                    break
        decoded = decoded.strip()

        print(f"\n{'='*60}")
        print(f"[{label}]")
        print(f"RESPONSE: {decoded}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
