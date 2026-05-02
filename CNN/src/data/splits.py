import os

import pandas as pd

from sklearn.model_selection import train_test_split

from dataset import collect_image_paths, DATASET_DIR, SPLIT_DIR
from src.configs import SEED

def make_splits():
    full_df = collect_image_paths(DATASET_DIR)
    print("Total images:", len(full_df))
    print(full_df["class_name"].value_counts())

    ixi_df = full_df[full_df["source"] == "IXI"].copy()
    other_df = full_df[full_df["source"] != "IXI"].copy()

    # Split non-IXI normally by label ( labels are no and yes, no says there is no cancer and yes the opposite )
    other_train, other_temp = train_test_split(
        other_df,
        test_size=0.30,
        stratify=other_df["label"],
        random_state=SEED
    )

    other_val, other_test = train_test_split(
        other_temp,
        test_size=0.50,
        stratify=other_temp["label"],
        random_state=SEED
    )

    # Split IXI by subject_id
    ixi_subjects = ixi_df[["subject_id"]].drop_duplicates()

    ixi_train_subjects, ixi_temp_subjects = train_test_split(
        ixi_subjects,
        test_size=0.30,
        random_state=SEED
    )

    ixi_val_subjects, ixi_test_subjects = train_test_split(
        ixi_temp_subjects,
        test_size=0.50,
        random_state=SEED
    )

    ixi_train = ixi_df[ixi_df["subject_id"].isin(ixi_train_subjects["subject_id"])]
    ixi_val = ixi_df[ixi_df["subject_id"].isin(ixi_val_subjects["subject_id"])]
    ixi_test = ixi_df[ixi_df["subject_id"].isin(ixi_test_subjects["subject_id"])]

    # Merge splits
    train_df = pd.concat([other_train, ixi_train], ignore_index=True)
    val_df = pd.concat([other_val, ixi_val], ignore_index=True)
    test_df = pd.concat([other_test, ixi_test], ignore_index=True)

    # saving CSVs
    train_df.to_csv(os.path.join(SPLIT_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(SPLIT_DIR, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(SPLIT_DIR, "test_split.csv"), index=False)

    # shuffle rows
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return train_df, val_df, test_df