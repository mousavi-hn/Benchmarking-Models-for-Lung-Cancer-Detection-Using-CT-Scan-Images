import os

from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

import src.configs as cfg

DATASET_DIR = "../data/MRI_Images"          # yes / no / IXI_no
CLASSICAL_MODEL_DIR = "../results/MRI_cnn_benchmark_results/saved_models"
OUTPUT_DIR = "../results/MRI_hybrid_benchmark_results"
SPLIT_DIR = os.path.join(OUTPUT_DIR, "splits")
MODEL_DIR = os.path.join(OUTPUT_DIR, "saved_models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

for folder in [OUTPUT_DIR, SPLIT_DIR, MODEL_DIR, PLOT_DIR, REPORT_DIR]:
    os.makedirs(folder, exist_ok=True)

def collect_image_paths(dataset_dir):
    records = []

    class_map = {
        "no": 0,
        "yes": 1,
    }

    for class_name, label in class_map.items():
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Folder not found: {class_dir}")

        for root, _, files in os.walk(class_dir):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in cfg.VALID_EXTENSIONS:
                    records.append({
                        "filepath": os.path.join(root, file),
                        "label": label,
                        "class_name": class_name,
                        "source": "non_IXI",
                        "subject_id": None
                    })

    ixi_dir = os.path.join(dataset_dir, "IXI_no")
    if not os.path.exists(ixi_dir):
        raise FileNotFoundError(f"Folder not found: {ixi_dir}")

    for root, _, files in os.walk(ixi_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in cfg.VALID_EXTENSIONS:
                subject_id = Path(root).name
                records.append({
                    "filepath": os.path.join(root, file),
                    "label": 0,
                    "class_name": "no",
                    "source": "IXI",
                    "subject_id": subject_id
                })

    full_df = pd.DataFrame(records)
    if full_df.empty:
        raise ValueError("No images found. Check dataset path and file extensions.")

    return full_df

def read_image(path, target_size):
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img, dtype=np.float32)
    return arr