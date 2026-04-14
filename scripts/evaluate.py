import os
import json
import yaml
import argparse
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report

from inference import IntentClassification


def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    parser.add_argument("--test_path", type=str, default="sample_data/processed/test.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading inference model...")
    classifier = IntentClassification(args.config)

    print("Loading test data...")
    test_df = pd.read_csv(args.test_path)

    required_cols = {"text", "label"}
    missing = required_cols - set(test_df.columns)
    if missing:
        raise ValueError(f"Test file thiếu cột bắt buộc: {missing}")

    y_true = []
    y_pred = []
    rows = []

    print("Running evaluation on test set...")
    for idx, row in test_df.iterrows():
        text = str(row["text"]).strip()
        true_label = str(row["label"]).strip()

        pred_label = classifier(text)

        y_true.append(true_label)
        y_pred.append(pred_label)

        rows.append({
            "text": text,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": true_label == pred_label,
        })

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} samples...")

    acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, zero_division=0)

    print("\n===== TEST RESULT =====")
    print(f"Accuracy: {acc:.4f}")
    print(report_text)

    predictions_df = pd.DataFrame(rows)
    predictions_path = os.path.join(args.output_dir, "test_predictions.csv")
    wrong_path = os.path.join(args.output_dir, "test_wrong_predictions.csv")
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    report_txt_path = os.path.join(args.output_dir, "test_report.txt")

    predictions_df.to_csv(predictions_path, index=False)
    predictions_df[predictions_df["correct"] == False].to_csv(wrong_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "classification_report": report_dict,
                "num_samples": len(test_df),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report_text)

    print("\nSaved files:")
    print(f"- {predictions_path}")
    print(f"- {wrong_path}")
    print(f"- {metrics_path}")
    print(f"- {report_txt_path}")


if __name__ == "__main__":
    main()