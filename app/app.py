"""
app.py
Gradio inference app for child grooming conversation classification.

Usage:
    python app/app.py

Requires one of:
    A) models/lora_adapter/  — LoRA adapter (loaded via transformers + peft)
    B) models/merged_gguf/   — GGUF file (loaded via llama-cpp-python)
    C) Set USE_GGUF=1 env var to force GGUF backend.

Interface:
    - Paste a multi-turn conversation (format: "username: message text")
    - Click Analyze
    - See risk level, highlighted suspicious lines, and explanation
"""

import json
import os
import re
import html
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent.parent
def _find_latest_checkpoint():
    adapter_dir = BASE / "models" / "lora_adapter"
    if not adapter_dir.exists():
        return None
    checkpoints = sorted(
        [d for d in adapter_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    return checkpoints[-1] if checkpoints else None

LORA_ADAPTER = _find_latest_checkpoint()
GGUF_DIR = BASE / "models" / "merged_gguf"
MERGED_DIR = BASE / "models" / "merged"
FALLBACK_BASE_MODEL = "Qwen/Qwen3.5-0.8B"

USE_GGUF = os.environ.get("USE_GGUF", "0") == "1"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a child safety AI assistant trained to detect grooming behavior "
    "in online conversations. Analyze conversations carefully and identify "
    "manipulation tactics, boundary violations, age-inappropriate content, "
    "trust-building patterns, isolation attempts, and any other indicators of "
    "predatory behavior targeting minors."
)

ANALYSIS_PROMPT_TEMPLATE = """\
Analyze the following conversation excerpt for signs of child grooming \
or sexual predatory behavior. Consider manipulation tactics, boundary \
violations, age-inappropriate content, trust-building patterns, and isolation attempts.

<conversation>
{conversation}
</conversation>"""


# ---------------------------------------------------------------------------
# Model backend
# ---------------------------------------------------------------------------

class GGUFBackend:
    def __init__(self, gguf_path: str):
        from llama_cpp import Llama
        print(f"[+] Loading GGUF model from {gguf_path} ...")
        self.llm = Llama(
            model_path=gguf_path,
            n_ctx=2048,
            n_gpu_layers=-1,   # offload all layers to GPU
            verbose=False,
        )
        print("[+] GGUF model loaded.")

    def generate(self, conversation: str, max_tokens: int = 512) -> str:
        prompt_text = ANALYSIS_PROMPT_TEMPLATE.format(conversation=conversation)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9,
        )
        return result["choices"][0]["message"]["content"]


class TransformersBackend:
    def __init__(self, model_dir: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print(f"[+] Loading tokenizer and model from {model_dir} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = Path(model_dir) if not model_dir.startswith("unsloth/") and not model_dir.startswith("Qwen/") else None
        is_hf_hub = model_path is None

        if is_hf_hub or (model_path / "config.json").exists():
            # Full model (HF hub or merged local)
            base = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto",
            )
            self.model = base
        else:
            # LoRA adapter — load base then attach
            base = AutoModelForCausalLM.from_pretrained(
                FALLBACK_BASE_MODEL,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(base, model_dir)

        self.model.eval()
        print(f"[+] Model loaded on {self.device}.")

    def generate(self, conversation: str, max_tokens: int = 512) -> str:
        import torch
        prompt_text = ANALYSIS_PROMPT_TEMPLATE.format(conversation=conversation)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=False)
        # Strip at the first end-of-turn or EOS token
        for stop in ["<|im_end|>", "</s>", "<|endoftext|>"]:
            if stop in decoded:
                decoded = decoded[:decoded.index(stop)]
        return decoded.strip()


def _has_valid_adapter(path: Path) -> bool:
    """Return True only if the directory contains a fully saved LoRA adapter."""
    return (path / "adapter_config.json").exists()


def load_backend():
    """Auto-detect and load the best available backend."""
    if USE_GGUF:
        gguf_files = list(GGUF_DIR.glob("*.gguf")) if GGUF_DIR.exists() else []
        if not gguf_files:
            raise FileNotFoundError(
                f"No .gguf files found in {GGUF_DIR}. "
                "Run scripts/finetune.py with EXPORT_GGUF=1 first."
            )
        return GGUFBackend(str(gguf_files[0]))

    # Prefer merged model, then LoRA adapter
    if MERGED_DIR.exists() and (MERGED_DIR / "config.json").exists():
        print("[+] Using merged fine-tuned model.")
        return TransformersBackend(str(MERGED_DIR))
    if LORA_ADAPTER and LORA_ADAPTER.exists() and _has_valid_adapter(LORA_ADAPTER):
        print("[+] Using LoRA fine-tuned adapter.")
        return TransformersBackend(str(LORA_ADAPTER))

    # Check for GGUF as fallback
    gguf_files = list(GGUF_DIR.glob("*.gguf")) if GGUF_DIR.exists() else []
    if gguf_files:
        return GGUFBackend(str(gguf_files[0]))

    # Fall back to base model (no fine-tuning yet)
    print(f"[!] No fine-tuned model found. Loading base model: {FALLBACK_BASE_MODEL}")
    print("[!] Run scripts/finetune.py to fine-tune. Using base model for now.")
    return TransformersBackend(FALLBACK_BASE_MODEL)


# ---------------------------------------------------------------------------
# Analysis logic
# ---------------------------------------------------------------------------

def parse_risk_level(response: str) -> str:
    upper = response.upper()
    if "GROOMING DETECTED" in upper:
        return "HIGH RISK"
    if "NO GROOMING DETECTED" in upper:
        return "LOW RISK"
    # fallback: legacy format
    if "HIGH RISK" in upper:
        return "HIGH RISK"
    if "MEDIUM RISK" in upper:
        return "MEDIUM RISK"
    return "LOW RISK"


RISK_COLORS = {
    "HIGH RISK": "#ff4444",
    "MEDIUM RISK": "#ff8800",
    "LOW RISK": "#22aa22",
}

RISK_BG = {
    "HIGH RISK": "#fff0f0",
    "MEDIUM RISK": "#fff8f0",
    "LOW RISK": "#f0fff0",
}


def highlight_conversation(conversation: str, response: str) -> str:
    """
    Build an HTML representation of the conversation where lines quoted
    in the model response are highlighted in red/orange.
    """
    lines = conversation.strip().splitlines()
    # Extract quoted fragments from the model response
    quoted = re.findall(r"\[([^\]]{10,})\]", response)
    quoted_lower = [q.lower().strip() for q in quoted]

    html_lines = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        safe_line = html.escape(line_stripped)
        # Check if this line (or a substantial part) was flagged
        flagged = any(
            q in line_stripped.lower() or line_stripped.lower() in q
            for q in quoted_lower
        )
        if flagged:
            html_lines.append(
                f'<div style="background:#ffe0e0;border-left:4px solid #ff4444;'
                f'padding:4px 8px;margin:2px 0;border-radius:3px;">'
                f'⚠️ {safe_line}</div>'
            )
        else:
            html_lines.append(
                f'<div style="padding:4px 8px;margin:2px 0;">{safe_line}</div>'
            )

    return "<div style='font-family:monospace;font-size:14px;'>" + "".join(html_lines) + "</div>"


_EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "examples.json")
with open(_EXAMPLES_PATH, encoding="utf-8") as _f:
    _EXAMPLES = json.load(_f)

def analyze_conversation(conversation: str, backend) -> tuple[str, str, str]:
    """
    Returns: (risk_badge_html, highlighted_conversation_html, raw_analysis_text)
    """
    conversation = conversation.strip()
    if not conversation:
        return "<p>Please enter a conversation.</p>", "", ""

    response = backend.generate(conversation)
    risk = parse_risk_level(response)
    color = RISK_COLORS[risk]
    bg = RISK_BG[risk]

    risk_badge = (
        f'<div style="background:{bg};border:2px solid {color};border-radius:8px;'
        f'padding:16px;text-align:center;">'
        f'<span style="font-size:24px;font-weight:bold;color:{color};">{risk}</span>'
        f'</div>'
    )

    highlighted = highlight_conversation(conversation, response)
    return risk_badge, highlighted, response


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui(backend):
    with gr.Blocks(title="IMPULSE — Child Grooming Detector") as demo:
        gr.Markdown(
            """
# IMPULSE — Child Grooming Conversation Detector
**For law enforcement, child safety researchers, and content moderation teams.**

Paste a multi-turn conversation below (format: `username: message`).
The model will classify the conversation and highlight suspicious exchanges.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                conv_input = gr.Textbox(
                    label="Conversation Input",
                    placeholder="alice: hey how old are you?\nbob: im 13\nalice: cool, are you home alone?",
                    lines=15,
                    max_lines=30,
                )
                analyze_btn = gr.Button("Analyze Conversation", variant="primary")

            with gr.Column(scale=1):
                risk_output = gr.HTML(label="Risk Assessment")
                highlighted_output = gr.HTML(label="Highlighted Conversation")

        with gr.Accordion("Full Model Analysis", open=False):
            raw_output = gr.Textbox(label="Raw Model Response", lines=10, interactive=False)

        gr.Examples(
            examples=[[ex["conversation"]] for ex in _EXAMPLES],
            inputs=[conv_input],
            label="Example Conversations",
        )

        def run_analysis(conv_text):
            risk_html, highlighted_html, raw = analyze_conversation(conv_text, backend)
            return risk_html, highlighted_html, raw

        analyze_btn.click(
            fn=run_analysis,
            inputs=[conv_input],
            outputs=[risk_output, highlighted_output, raw_output],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[+] Loading model backend ...")
    backend = load_backend()

    print("[+] Launching Gradio app ...")
    demo = build_ui(backend)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
