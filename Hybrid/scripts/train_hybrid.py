import os
os.environ["KERAS_BACKEND"] = "jax"

import warnings
warnings.filterwarnings("ignore")

import random

import pandas as pd
import numpy as np

import keras

from src.data.loader import OUTPUT_DIR, collect_image_paths, DATASET_DIR
from src.data.splits import  make_splits
from src.data.dataset import  make_generators
from src.models.classical import  find_classical_model_path
from src.train.trainer import train_one_hybrid

import src.configs as cfg

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
keras.utils.set_random_seed(cfg.SEED)


def main():
    full_df = collect_image_paths(DATASET_DIR)
    print("Total images:", len(full_df))
    print(full_df["class_name"].value_counts())

    train_df, val_df, test_df = make_splits(full_df)

    print("\nTrain label counts:")
    print(train_df["class_name"].value_counts())
    print("\nVal label counts:")
    print(val_df["class_name"].value_counts())
    print("\nTest label counts:")
    print(test_df["class_name"].value_counts())

    all_results = []

    model_names = cfg.MODEL_NAMES

    for model_name in model_names:
        try:
            preprocess_func = cfg.MODEL_CONFIGS[model_name]["preprocess"]
            model_path = find_classical_model_path(model_name)

            print(f"\nUsing saved classical model: {model_path}")

            train_seq, val_seq, test_seq = make_generators(
                preprocess_func=preprocess_func,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df
            )

            qubit_list = cfg.QUANTUM_QUBITS

            for n_qubits in qubit_list:
                try:
                    result = train_one_hybrid(
                        model_name=model_name,
                        model_path=model_path,
                        train_seq=train_seq,
                        val_seq=val_seq,
                        test_seq=test_seq,
                        n_qubits=n_qubits,
                        q_depth=cfg.Q_DEPTH
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"\nHybrid run failed for {model_name} with {n_qubits} qubits:")
                    print(str(e))

        except Exception as e:
            print(f"\nModel {model_name} failed before hybrid training:")
            print(str(e))

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df["balanced_accuracy"] = (
            results_df["recall_sensitivity"] + results_df["specificity"]
        ) / 2.0

        results_df = results_df.sort_values(
            by=["roc_auc", "f1_score", "balanced_accuracy", "accuracy"],
            ascending=False
        ).reset_index(drop=True)

        summary_path = os.path.join(OUTPUT_DIR, "hybrid_benchmark_summary.csv")
        results_df.to_csv(summary_path, index=False)

        print("\nFinal ranking:")
        print(results_df[
            [
                "model_name",
                "n_qubits",
                "q_depth",
                "accuracy",
                "precision",
                "recall_sensitivity",
                "specificity",
                "f1_score",
                "roc_auc",
                "balanced_accuracy",
                "tp", "tn", "fp", "fn",
                "training_time_sec"
            ]
        ])

        print(f"\nSummary saved to: {summary_path}")
    else:
        print("No hybrid model finished successfully.")

if __name__ == "__main__":
    main()