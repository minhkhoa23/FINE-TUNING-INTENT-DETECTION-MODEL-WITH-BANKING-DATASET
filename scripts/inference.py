import os
import re
import json
import yaml
import argparse
from difflib import get_close_matches
from typing import Dict, List, Any

import torch
from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "You are a banking intent classification assistant.\n"
    "Given one customer message, output only the intent label name.\n"
    "Do not explain. Do not add extra words."
)


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label_text(text: str) -> str:
    """
    Chuẩn hóa text model sinh ra để dễ match với label.
    """
    text = text.strip().lower()
    text = text.replace("`", "")
    text = text.replace('"', "")
    text = text.replace("'", "")
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ", "_")
    return text


class IntentClassification:
    def __init__(self, model_path: str):
        """
        model_path: đường dẫn tới file config inference.yaml
        """
        self.config = load_yaml(model_path)

        self.model_dir = self.config["model_dir"]
        self.id2label_path = self.config["id2label_path"]
        self.max_seq_length = int(self.config.get("max_seq_length", 512))
        self.load_in_4bit = bool(self.config.get("load_in_4bit", True))
        self.max_new_tokens = int(self.config.get("max_new_tokens", 16))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        id2label_raw = load_json(self.id2label_path)
        self.id2label = {int(k): v for k, v in id2label_raw.items()}
        self.valid_labels: List[str] = list(self.id2label.values())
        self.valid_labels_normalized = [normalize_label_text(x) for x in self.valid_labels]

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_dir,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)

        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError(
                "Tokenizer does not support apply_chat_template. "
                "Please update unsloth / transformers."
            )

    def _build_messages(self, message: str):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]

    def _postprocess_prediction(self, generated_text: str) -> str:
        """
        Cố gắng ép output của model về đúng 1 label hợp lệ.
        """
        raw = generated_text.strip()
        norm = normalize_label_text(raw)

        # 1) match exact
        if norm in self.valid_labels_normalized:
            idx = self.valid_labels_normalized.index(norm)
            return self.valid_labels[idx]

        # 2) tìm label xuất hiện bên trong output
        for original, normalized in zip(self.valid_labels, self.valid_labels_normalized):
            if normalized in norm:
                return original

        # 3) fuzzy matching
        matches = get_close_matches(norm, self.valid_labels_normalized, n=1, cutoff=0.5)
        if matches:
            best = matches[0]
            idx = self.valid_labels_normalized.index(best)
            return self.valid_labels[idx]

        # 4) fallback: trả raw nếu không match được gì
        return raw

    def __call__(self, message: str) -> str:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Input message must be a non-empty string.")

        messages = self._build_messages(message.strip())

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        predicted_label = self._postprocess_prediction(generated_text)
        return predicted_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    parser.add_argument("--text", type=str, required=True, help="Input message for intent prediction")
    args = parser.parse_args()

    classifier = IntentClassification(args.config)
    predicted_label = classifier(args.text)

    print("Input:", args.text)
    print("Predicted label:", predicted_label)


if __name__ == "__main__":
    main()