import os
import re
import json
import argparse
from typing import Optional, List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_text(text: str, lowercase: bool = True) -> str:
    """
    Làm sạch text cơ bản:
    - bỏ khoảng trắng thừa
    - lowercase nếu cần
    """
    if pd.isna(text):
        text = ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    if lowercase:
        text = text.lower()

    return text


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    df = pd.read_csv(path)

    required_cols = {"text", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"File {path} thiếu cột bắt buộc: {missing}")

    df = df.rename(columns={"category": "label"})
    df = df[["text", "label"]].copy()
    return df


def select_subset_labels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_labels: Optional[int] = None,
    label_list: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Chọn subset intent theo 2 cách:
    1. label_list được chỉ định rõ
    2. num_labels: lấy N label đầu tiên theo thứ tự alphabet
    """
    train_labels = set(train_df["label"].unique())
    test_labels = set(test_df["label"].unique())
    common_labels = sorted(list(train_labels.intersection(test_labels)))

    if label_list is not None and len(label_list) > 0:
        selected_labels = [label for label in label_list if label in common_labels]
        if not selected_labels:
            raise ValueError("Không có label hợp lệ nào trong --label_list.")
    elif num_labels is not None:
        if num_labels <= 0:
            raise ValueError("--num_labels phải > 0")
        selected_labels = common_labels[:num_labels]
    else:
        selected_labels = common_labels

    train_df = train_df[train_df["label"].isin(selected_labels)].copy()
    test_df = test_df[test_df["label"].isin(selected_labels)].copy()

    return train_df, test_df, selected_labels


def build_label_mapping(labels: List[str]) -> tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted(labels)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def apply_label_mapping(df: pd.DataFrame, label2id: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["label_id"] = df["label"].map(label2id)
    if df["label_id"].isnull().any():
        bad_rows = df[df["label_id"].isnull()]
        raise ValueError(f"Có label không map được:\n{bad_rows.head()}")
    df["label_id"] = df["label_id"].astype(int)
    return df


def sample_per_label(
    df: pd.DataFrame,
    max_samples_per_label: Optional[int],
    random_state: int,
) -> pd.DataFrame:
    """
    Giới hạn số lượng mẫu tối đa trên mỗi label nếu cần.
    """
    if max_samples_per_label is None:
        return df

    if max_samples_per_label <= 0:
        raise ValueError("--max_samples_per_label phải > 0")

    sampled_parts = []
    for label_name, group in df.groupby("label", sort=True):
        n = min(len(group), max_samples_per_label)
        sampled_group = group.sample(n=n, random_state=random_state)
        sampled_parts.append(sampled_group)

    out_df = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=random_state)
    out_df = out_df.reset_index(drop=True)
    return out_df


def save_json(data, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Preprocess BANKING77 dataset from local CSV files.")
    parser.add_argument("--train_path", type=str, default="sample_data/train.csv")
    parser.add_argument("--test_path", type=str, default="sample_data/test.csv")
    parser.add_argument("--output_dir", type=str, default="sample_data/processed")

    parser.add_argument("--num_labels", type=int, default=None,
                        help="Số intent muốn giữ lại. Ví dụ: 10")
    parser.add_argument("--label_list", type=str, default=None,
                        help='Danh sách label cách nhau bởi dấu phẩy. Ví dụ: "card_arrival,cash_withdrawal"')
    parser.add_argument("--max_samples_per_label", type=int, default=None,
                        help="Số mẫu tối đa cho mỗi label ở train/test")

    parser.add_argument("--create_val", action="store_true",
                        help="Có tách validation từ train hay không")
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="Tỷ lệ validation tách từ train")
    parser.add_argument("--lowercase", action="store_true",
                        help="Chuyển text về lowercase")
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Đang đọc dữ liệu...")
    train_df = load_csv(args.train_path)
    test_df = load_csv(args.test_path)

    print(f"Train gốc: {train_df.shape}")
    print(f"Test gốc : {test_df.shape}")

    label_list = None
    if args.label_list:
        label_list = [x.strip() for x in args.label_list.split(",") if x.strip()]

    print("Đang chọn subset label...")
    train_df, test_df, selected_labels = select_subset_labels(
        train_df=train_df,
        test_df=test_df,
        num_labels=args.num_labels,
        label_list=label_list,
    )

    print(f"Số label được chọn: {len(selected_labels)}")
    print("Danh sách label:")
    for i, label in enumerate(selected_labels):
        print(f"{i:02d}. {label}")

    print("Đang normalize text...")
    train_df["text"] = train_df["text"].apply(lambda x: normalize_text(x, lowercase=args.lowercase))
    test_df["text"] = test_df["text"].apply(lambda x: normalize_text(x, lowercase=args.lowercase))

    train_df = train_df.dropna(subset=["text", "label"]).drop_duplicates().reset_index(drop=True)
    test_df = test_df.dropna(subset=["text", "label"]).drop_duplicates().reset_index(drop=True)

    print("Đang sampling theo label nếu cần...")
    train_df = sample_per_label(train_df, args.max_samples_per_label, args.random_state)
    test_df = sample_per_label(test_df, args.max_samples_per_label, args.random_state)

    all_labels = sorted(train_df["label"].unique().tolist())
    label2id, id2label = build_label_mapping(all_labels)

    train_df = apply_label_mapping(train_df, label2id)
    test_df = apply_label_mapping(test_df, label2id)

    if args.create_val:
        print("Đang tách validation từ train...")

        train_split, val_split = train_test_split(
        train_df,
        test_size=0.1,
        random_state=args.random_state,
        stratify=train_df["label_id"],
        )   

        train_split = train_split.reset_index(drop=True)
        val_split = val_split.reset_index(drop=True)
    else:
        train_split = train_df.reset_index(drop=True)
        val_split = None

    test_df = test_df.reset_index(drop=True)

    train_out = os.path.join(args.output_dir, "train.csv")
    test_out = os.path.join(args.output_dir, "test.csv")
    labels_out = os.path.join(args.output_dir, "label2id.json")
    id2label_out = os.path.join(args.output_dir, "id2label.json")
    stats_out = os.path.join(args.output_dir, "stats.json")

    print("Đang lưu file...")
    train_split.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    if val_split is not None:
        val_out = os.path.join(args.output_dir, "val.csv")
        val_split.to_csv(val_out, index=False)
    else:
        val_out = None

    save_json(label2id, labels_out)
    save_json(id2label, id2label_out)

    stats = {
        "train_path": args.train_path,
        "test_path": args.test_path,
        "output_dir": args.output_dir,
        "num_selected_labels": len(selected_labels),
        "selected_labels": selected_labels,
        "train_size": len(train_split),
        "val_size": len(val_split) if val_split is not None else 0,
        "test_size": len(test_df),
        "lowercase": args.lowercase,
        "create_val": args.create_val,
        "val_size_ratio": args.val_size if args.create_val else None,
        "max_samples_per_label": args.max_samples_per_label,
        "random_state": args.random_state,
    }
    save_json(stats, stats_out)

    print("\nHoàn tất.")
    print(f"Train processed: {train_out}")
    if val_out is not None:
        print(f"Val processed  : {val_out}")
    print(f"Test processed : {test_out}")
    print(f"label2id       : {labels_out}")
    print(f"id2label       : {id2label_out}")
    print(f"stats          : {stats_out}")

    print("\nPhân bố train theo label:")
    print(train_split["label"].value_counts().sort_index())

    if val_split is not None:
        print("\nPhân bố val theo label:")
        print(val_split["label"].value_counts().sort_index())

    print("\nPhân bố test theo label:")
    print(test_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()