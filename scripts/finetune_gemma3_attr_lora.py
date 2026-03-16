import os
import sys
from pathlib import Path
import json
from typing import Dict, List

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


# ----------------------------
# Paths / project wiring
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import prompt + schema helpers from your existing script
from scripts.slm_attribute_labeler import (
    construct_prompt_optimized,
    ATTR_ATTRS,
    _winner_str,
)

DATA_PATH = PROJECT_ROOT / "data" / "phase3_slm_labeledgpt4omini.parquet"  # adjust if needed
OUTPUT_DIR = PROJECT_ROOT / "models" / "gemma3_attr_lora"
MODEL_NAME = "google/gemma-3-4b-it"

MAX_SEQ_LEN = 1024
BATCH_SIZE_PER_DEVICE = 2
GRAD_ACCUM_STEPS = 8
LR = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03


# ----------------------------
# Build text examples
# ----------------------------

def build_json_target(row: pd.Series) -> str:
    """
    Convert your per-attribute winner columns into the JSON the model should output.
    Uses the same schema as parse_slm_response / PROMPT_OPENAI_OPTIMIZED.
    """
    obj: Dict[str, str] = {}

    # Attribute winners
    for attr in ATTR_ATTRS:
        col = f"attr_{attr}_winner"
        if col in row:
            obj[attr] = _winner_str(str(row[col]).strip().lower())
        else:
            obj[attr] = "none"

    # Optional golden_label and reason if present
    if "golden_label" in row:
        gl = str(row["golden_label"]).strip().lower()
        if gl not in ("base", "alt", "abstain"):
            gl = "abstain"
        obj["golden_label"] = gl

    if "SLM_reason" in row and isinstance(row["SLM_reason"], str):
        obj["reason"] = row["SLM_reason"]

    return json.dumps(obj, ensure_ascii=False)


def row_to_example(row: pd.Series) -> Dict[str, str]:
    """
    One supervised example:
    - 'prompt': same optimized prompt your labeler uses
    - 'response': the JSON answer we want Gemma to generate
    """
    prompt = construct_prompt_optimized(row)
    target_json = build_json_target(row)

    # Turn into simple chat-style text: user prompt + assistant JSON
    # You can tweak this to use Gemma's chat template via tokenizer.apply_chat_template if you prefer.
    text = (
        "<start_of_turn>user\n"
        + prompt
        + "<end_of_turn>\n"
        "<start_of_turn>model\n"
        + target_json
        + "<end_of_turn>\n"
    )
    return {"text": text}


def load_dataset_from_parquet(path: Path) -> Dataset:
    df = pd.read_parquet(path)
    # Optionally filter to rows that actually have winners
    mask = df[[f"attr_{a}_winner" for a in ATTR_ATTRS if f"attr_{a}_winner" in df.columns]].notna().any(axis=1)
    df = df[mask].reset_index(drop=True)

    examples: List[Dict[str, str]] = [row_to_example(row) for _, row in df.iterrows()]
    return Dataset.from_list(examples)


# ----------------------------
# Tokenization
# ----------------------------

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding="max_length",
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4-bit quantization (QLoRA style)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # LoRA on attention + MLP projections
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build HF dataset
    ds = load_dataset_from_parquet(DATA_PATH)
    tokenized = ds.map(lambda ex: tokenize_function(ex, tokenizer), batched=True, remove_columns=["text"])
    tokenized = tokenized.with_format("torch")
    tokenized = tokenized.map(lambda ex: {"labels": ex["input_ids"]})

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="none",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()