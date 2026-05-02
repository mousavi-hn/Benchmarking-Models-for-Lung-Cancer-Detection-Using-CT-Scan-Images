import os
import random

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf

from src.train.trainer import train_and_evaluate
from src.data.splits import make_splits
from src.data.dataset import OUTPUT_DIR
from src.configs import SEED, MODEL_NAMES

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# RUN ALL MODELS
all_results = []

for model_name in MODEL_NAMES:
    train_df, val_df, test_df = make_splits()
    try:
        result = train_and_evaluate(model_name, train_df, val_df, test_df)
        all_results.append(result)
    except Exception as e:
        print(f"\nModel {model_name} failed with error:")
        print(str(e))

# SAVE FINAL COMPARISON TABLE
if all_results:
    results_df = pd.DataFrame(all_results)

    # Sort by ROC-AUC first, then F1, then recall
    results_df = results_df.sort_values(
        by=["roc_auc", "f1_score", "recall_sensitivity"],
        ascending=False
    ).reset_index(drop=True)

    results_df.to_csv(os.path.join(OUTPUT_DIR, "cnn_benchmark_summary.csv"), index=False)

    print("\nFinal ranking:")
    print(results_df[
        [
            "model_name",
            "accuracy",
            "precision",
            "recall_sensitivity",
            "specificity",
            "f1_score",
            "roc_auc",
            "tp", "tn", "fp", "fn",
            "training_time_sec"
        ]
    ])

else:
    print("No model finished successfully.")