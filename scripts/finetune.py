"""
finetune.py
LoRA fine-tuning of Qwen2.5-0.5B-Instruct using transformers + peft + trl.
Runs on CPU (slow but functional) or CUDA GPU automatically.

Usage:
    python scripts/finetune.py

Prerequisites:
    pip install -r requirements.txt

Outputs:
    models/lora_adapter/   — LoRA weights + tokenizer
    models/merged/         — merged model in safetensors (optional)
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent.parent
TRAIN_JSONL = BASE / "data" / "train.jsonl"
VAL_JSONL = BASE / "data" / "val.jsonl"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

LORA_OUTPUT = str(MODELS_DIR / "lora_adapter")
MERGED_OUTPUT = str(MODELS_DIR / "merged")
GGUF_OUTPUT = str(MODELS_DIR / "merged_gguf")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True    # QLoRA: 4-bit base weights, train LoRA adapters in bf16

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 10
EVAL_STEPS = 500
SAVE_STEPS = 500
FP16 = False
BF16 = True   # RTX 4080 (Ampere) supports bf16


# ---------------------------------------------------------------------------
# Imports (heavy — done after path setup)
# ---------------------------------------------------------------------------
def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load base model
    # ------------------------------------------------------------------
    print(f"[+] Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if LOAD_IN_4BIT and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
    model.config.use_cache = False

    # ------------------------------------------------------------------
    # 2. Attach LoRA adapters
    # ------------------------------------------------------------------
    print("[+] Attaching LoRA adapters ...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 3. Load dataset
    # ------------------------------------------------------------------
    print("[+] Loading dataset ...")
    if not TRAIN_JSONL.exists():
        raise FileNotFoundError(
            f"{TRAIN_JSONL} not found. Run scripts/prepare_dataset.py first."
        )

    raw = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_JSONL),
            "validation": str(VAL_JSONL),
        },
    )
    print(f"    Train: {len(raw['train'])} samples, Val: {len(raw['validation'])} samples")

    # ------------------------------------------------------------------
    # 4. Format samples using the chat template
    # ------------------------------------------------------------------
    def format_sample(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    raw = raw.map(format_sample, remove_columns=["messages"])

    # ------------------------------------------------------------------
    # 5. SFT Training
    # ------------------------------------------------------------------
    print("[+] Starting SFT training ...")

    training_args = SFTConfig(
        output_dir=LORA_OUTPUT,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        fp16=FP16,
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to="none",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=True,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=raw["train"],
        eval_dataset=raw["validation"].select(range(500)),
        args=training_args,
    )

    trainer_stats = trainer.train()
    print(f"[+] Training complete. Stats: {trainer_stats}")

    # ------------------------------------------------------------------
    # 6. Save LoRA adapter
    # ------------------------------------------------------------------
    print(f"[+] Saving LoRA adapter → {LORA_OUTPUT}")
    model.save_pretrained(LORA_OUTPUT)
    tokenizer.save_pretrained(LORA_OUTPUT)

    # ------------------------------------------------------------------
    # 7. (Optional) Save merged model
    # ------------------------------------------------------------------
    if os.environ.get("EXPORT_MERGED", "0") == "1":
        print("[+] Merging LoRA into base weights ...")
        merged = model.merge_and_unload()
        merged.save_pretrained(MERGED_OUTPUT)
        tokenizer.save_pretrained(MERGED_OUTPUT)
        print(f"    Merged model saved → {MERGED_OUTPUT}")

    print("[✓] Fine-tuning pipeline complete.")


if __name__ == "__main__":
    main()
