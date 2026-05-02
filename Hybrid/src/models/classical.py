import os

import keras
from keras import Model

from src.data.loader import CLASSICAL_MODEL_DIR

def find_classical_model_path(model_name):
    candidates = [
        os.path.join(CLASSICAL_MODEL_DIR, f"{model_name}.keras"),
        os.path.join(CLASSICAL_MODEL_DIR, f"{model_name}_best.keras"),
        os.path.join(CLASSICAL_MODEL_DIR, f"{model_name.lower()}.keras"),
        os.path.join(CLASSICAL_MODEL_DIR, f"{model_name.lower()}_best.keras"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"No .keras model found for {model_name} in {CLASSICAL_MODEL_DIR}"
    )

def load_feature_extractor(model_path):
    loaded_model = keras.models.load_model(model_path, compile=False)

    if len(loaded_model.layers) < 2:
        raise ValueError(f"Model at {model_path} does not have enough layers.")

    # We remove the last sigmoid layer and keep the penultimate representation
    feature_extractor = Model(
        inputs=loaded_model.input,
        outputs=loaded_model.layers[-2].output,
        name=f"{loaded_model.name}_feature_extractor"
    )
    return feature_extractor