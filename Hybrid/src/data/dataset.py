import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pylidc as pl

from src.configs import DATASET_DIR, IMG_SIZE, HU_MIN, HU_MAX

# PATHS AND SETTINGS
OUTPUT_DIR = "../results/CT_cnn_benchmark_results"
SPLIT_DIR = os.path.join(OUTPUT_DIR, "splits")
MODEL_DIR = os.path.join(OUTPUT_DIR, "saved_models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def normalize_hu(slice_img):
    slice_img = np.clip(slice_img, HU_MIN, HU_MAX)
    slice_img = (slice_img - HU_MIN) / (HU_MAX - HU_MIN)
    slice_img = (slice_img * 255).astype(np.uint8)
    return slice_img


def resize_img(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def get_positive_slice_indices(scan):
    positive_slices = set()
    nodules = scan.cluster_annotations()

    for nodule in nodules:
        for ann in nodule:
            bbox = ann.bbox()

            z_slice = bbox[2]
            z_min = z_slice.start
            z_max = z_slice.stop

            for z in range(z_min, z_max):
                positive_slices.add(z)

    return positive_slices


def export_scan(scan, metadata_rows):
    volume = scan.to_volume()  # shape: H, W, Z
    positive_slices = get_positive_slice_indices(scan)

    patient_id = scan.patient_id
    scan_id = scan.id

    for z in range(volume.shape[2]):
        label = "yes" if z in positive_slices else "no"

        img = volume[:, :, z]
        img = normalize_hu(img)
        img = resize_img(img)

        out_dir = OUTPUT_DIR / "all_images" / label
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{patient_id}_scan{scan_id}_slice{z:04d}.png"
        out_path = out_dir / filename

        cv2.imwrite(str(out_path), img)

        metadata_rows.append({
            "patient_id": patient_id,
            "scan_id": scan_id,
            "slice_index": z,
            "label": label,
            "file_path": str(out_path)
        })


def generate_metadata():
    if Path(DATASET_DIR / "metadata_all_slices.csv").exists():
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    scans = pl.query(pl.Scan).all()
    metadata_rows = []

    print(f"Found {len(scans)} scans.")

    for scan in tqdm(scans, desc="Exporting scans"):
        try:
            export_scan(scan, metadata_rows)
        except Exception as e:
            print(f"Failed scan {scan.id}, patient {scan.patient_id}: {e}")

    df = pd.DataFrame(metadata_rows)
    df.to_csv(DATASET_DIR / "metadata_all_slices.csv", index=False)

    print("Done.")
    print(df["label"].value_counts())