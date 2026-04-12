import os
import json
import yaml
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch
import pandas as pd
from datasets import Dataset

from unsloth import FastVisionModel, is_bf16_supported
from trl import SFTTrainer, SFTConfig


SYSTEM_PROMPT = (
    "You are a banking intent classification assistant.\n"
    "Given one customer message, output only the intent label name.\n"
    "Do not explain. Do not add extra words."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_exists(path: str, what: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def read_csv_required(path: str) -> pd.DataFrame:
    ensure_exists(path, "CSV file")
    df = pd.read_csv(path)
    required_cols = {"text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def load_id2label(path: Optional[str], train_df: pd.DataFrame) -> Dict[int, str]:
    """
    Ưu tiên dùng file id2label.json sinh từ preprocess_data.py.
    Nếu không có thì tự tạo từ label text theo thứ tự alphabet.
    """
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # normalize key to int
        return {int(k): v for k, v in data.items()}

    labels = sorted(train_df["label"].dropna().unique().tolist())
    return {i: label for i, label in enumerate(labels)}


def validate_labels(train_df: pd.DataFrame, val_df: Optional[pd.DataFrame], test_df: Optional[pd.DataFrame]) -> None:
    train_labels = set(train_df["label"].unique())
    if val_df is not None:
        missing = set(val_df["label"].unique()) - train_labels
        if missing:
            raise ValueError(f"Validation has labels not seen in train: {sorted(missing)}")
    if test_df is not None:
        missing = set(test_df["label"].unique()) - train_labels
        if missing:
            raise ValueError(f"Test has labels not seen in train: {sorted(missing)}")


def row_to_messages(text: str, label: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
        {"role": "assistant", "content": label},
    ]


def build_hf_dataset(df: pd.DataFrame) -> Dataset:
    records = []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        label = str(row["label"]).strip()
        if not text or not label:
            continue
        records.append(
            {
                "messages": row_to_messages(text, label),
                "text": text,
                "label": label,
            }
        )
    return Dataset.from_list(records)


def maybe_prepare_chat_template(tokenizer):
    """
    Một số tokenizer đã có sẵn chat template.
    Nếu tokenizer.apply_chat_template không chạy được, báo lỗi rõ ràng.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            "Tokenizer does not support apply_chat_template. "
            "Please update transformers / unsloth and verify the Gemma 4 tokenizer."
        )
    return tokenizer


def format_example_for_sft(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def compute_steps(
    train_size: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
) -> int:
    effective_batch = max(1, per_device_train_batch_size * gradient_accumulation_steps)
    steps_per_epoch = math.ceil(train_size / effective_batch)
    return max(1, int(steps_per_epoch * num_train_epochs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    train_path = cfg["train_path"]
    val_path = cfg.get("val_path")
    test_path = cfg.get("test_path")
    id2label_path = cfg.get("id2label_path")

    model_name = cfg.get("model_name", "unsloth/gemma-4-e2b-it")
    output_dir = cfg.get("output_dir", "outputs/gemma4_e2b_lora")
    os.makedirs(output_dir, exist_ok=True)

    max_seq_length = int(cfg.get("max_seq_length", 512))
    load_in_4bit = bool(cfg.get("load_in_4bit", True))
    per_device_train_batch_size = int(cfg.get("per_device_train_batch_size", 2))
    per_device_eval_batch_size = int(cfg.get("per_device_eval_batch_size", 2))
    gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", 4))
    learning_rate = float(cfg.get("learning_rate", 2e-4))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.03))
    num_train_epochs = float(cfg.get("num_train_epochs", 3))
    logging_steps = int(cfg.get("logging_steps", 10))
    save_steps = int(cfg.get("save_steps", 100))
    eval_steps = int(cfg.get("eval_steps", 100))
    lr_scheduler_type = cfg.get("lr_scheduler_type", "cosine")
    optim = cfg.get("optim", "adamw_8bit")

    lora_r = int(cfg.get("lora_r", 16))
    lora_alpha = int(cfg.get("lora_alpha", 16))
    lora_dropout = float(cfg.get("lora_dropout", 0.0))
    use_rslora = bool(cfg.get("use_rslora", False))
    random_state = int(cfg.get("random_state", seed))

    fp16 = not is_bf16_supported()
    bf16 = is_bf16_supported()

    print("Loading processed CSV files...")
    train_df = read_csv_required(train_path)
    val_df = read_csv_required(val_path) if val_path else None
    test_df = read_csv_required(test_path) if test_path else None

    validate_labels(train_df, val_df, test_df)
    id2label = load_id2label(id2label_path, train_df)
    label_set = sorted(train_df["label"].unique().tolist())

    print(f"Train size: {len(train_df)}")
    print(f"Val size  : {len(val_df) if val_df is not None else 0}")
    print(f"Test size : {len(test_df) if test_df is not None else 0}")
    print(f"Num labels: {len(label_set)}")

    print("Loading Gemma 4 E2B with Unsloth...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    tokenizer = maybe_prepare_chat_template(tokenizer)

    # E2B/E4B nên khởi đầu bằng text-focused LoRA, không fine-tune vision layers
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_rslora=use_rslora,
        random_state=random_state,
    )

    FastVisionModel.for_training(model)

    print("Converting CSV to chat-format dataset...")
    train_ds = build_hf_dataset(train_df)
    val_ds = build_hf_dataset(val_df) if val_df is not None else None

    train_ds = train_ds.map(
        lambda x: format_example_for_sft(x, tokenizer),
        remove_columns=train_ds.column_names,
        desc="Formatting train with chat template",
    )
    if val_ds is not None:
        val_ds = val_ds.map(
            lambda x: format_example_for_sft(x, tokenizer),
            remove_columns=val_ds.column_names,
            desc="Formatting val with chat template",
        )

    total_steps = compute_steps(
        train_size=len(train_ds),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
    )
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    print("Building trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=output_dir,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            lr_scheduler_type=lr_scheduler_type,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if val_ds is not None else None,
            eval_strategy="steps" if val_ds is not None else "no",
            save_strategy="steps",
            optim=optim,
            fp16=fp16,
            bf16=bf16,
            report_to=cfg.get("report_to", "none"),
            seed=seed,
            remove_unused_columns=False,
            packing=bool(cfg.get("packing", False)),
        ),
    )

    print("Start training...")
    train_result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Lưu thêm metadata để inference dùng lại
    with open(os.path.join(output_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "train_config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    metrics = train_result.metrics
    with open(os.path.join(output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if val_ds is not None:
        print("Evaluating on validation set...")
        eval_metrics = trainer.evaluate()
        with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
        print(eval_metrics)

    print("Done.")
    print(f"Saved LoRA adapter + tokenizer to: {output_dir}")


if __name__ == "__main__":
    main()