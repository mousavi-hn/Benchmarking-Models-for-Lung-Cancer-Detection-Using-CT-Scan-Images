import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataset import generate_metadata

from src.configs import DATASET_DIR


RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def copy_images(df, split_name):
    for _, row in df.iterrows():
        src = Path(row["file_path"])
        label = row["label"]

        dst_dir = OUTPUT_DIR / "dataset" / split_name / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst = dst_dir / src.name
        shutil.copy2(src, dst)


def balance_negatives(df):
    yes_df = df[df["label"] == "yes"]
    no_df = df[df["label"] == "no"]

    n_yes = len(yes_df)

    if len(no_df) > n_yes:
        no_df = no_df.sample(n=n_yes, random_state=RANDOM_STATE)

    return pd.concat([yes_df, no_df]).sample(frac=1, random_state=RANDOM_STATE)


def make_splits():
    generate_metadata()
    metadata_path = DATASET_DIR / "metadata_all_slices.csv"
    df = pd.read_csv(metadata_path)

    scan_labels = (
        df.groupby("scan_id")["label"]
        .apply(lambda x: "yes" if "yes" in set(x) else "no")
        .reset_index()
    )

    train_scans, temp_scans = train_test_split(
        scan_labels["scan_id"],
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=scan_labels["label"]
    )

    temp_df = scan_labels[scan_labels["scan_id"].isin(temp_scans)]

    val_scans, test_scans = train_test_split(
        temp_df["scan_id"],
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=temp_df["label"]
    )

    train_df = df[df["scan_id"].isin(train_scans)]
    val_df = df[df["scan_id"].isin(val_scans)]
    test_df = df[df["scan_id"].isin(test_scans)]

    train_df = balance_negatives(train_df)
    val_df = balance_negatives(val_df)
    test_df = balance_negatives(test_df)

    copy_images(train_df, "train")
    copy_images(val_df, "val")
    copy_images(test_df, "test")

    train_df.to_csv(OUTPUT_DIR / "metadata_train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "metadata_val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "metadata_test.csv", index=False)

    print("Train:")
    print(train_df["label"].value_counts())

    print("\nVal:")
    print(val_df["label"].value_counts())

    print("\nTest:")
    print(test_df["label"].value_counts())

    return train_df, val_df, test_df