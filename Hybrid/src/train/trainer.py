import os
import time
import json

import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.data.loader import MODEL_DIR, REPORT_DIR
from src.models.hybrid import build_hybrid_model, unfreeze_top_fraction, QuantumLayer
from src.evaluate.metrics import calculate_metrics, predict_on_sequence
from src.evaluate.plots import plot_history
from src.configs import LEARNING_RATE_HEAD, LEARNING_RATE_FINE, EPOCHS_HEAD, EPOCHS_FINE

def train_one_hybrid(model_name, model_path, train_seq, val_seq, test_seq, n_qubits, q_depth=2):
    model_tag = f"{model_name}_hybrid_{n_qubits}q_d{q_depth}"
    best_model_path = os.path.join(MODEL_DIR, f"{model_tag}.keras")

    print("\n" + "=" * 70)
    print(f"Training {model_tag}")
    print("=" * 70)

    hybrid_model, feature_extractor = build_hybrid_model(
        model_path=model_path,
        n_qubits=n_qubits,
        q_depth=q_depth,
        freeze_backbone=True
    )

    hybrid_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_head = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1),
        ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    start_time = time.time()

    history_head = hybrid_model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks_head,
        verbose=1
    )

    # Fine-tuning phase
    unfreeze_top_fraction(feature_extractor, fraction=0.30)

    hybrid_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_fine = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1),
        ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    history_fine = hybrid_model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS_FINE,
        callbacks=callbacks_fine,
        verbose=1
    )

    total_training_time = time.time() - start_time

    custom_objects = {"QuantumLayer": QuantumLayer}
    best_model = keras.models.load_model(
        best_model_path,
        compile=False,
        custom_objects=custom_objects
    )

    y_true, y_prob = predict_on_sequence(best_model, test_seq)

    metrics = calculate_metrics(y_true, y_prob, threshold=0.5)
    metrics["model_name"] = model_name
    metrics["n_qubits"] = int(n_qubits)
    metrics["q_depth"] = int(q_depth)
    metrics["training_time_sec"] = float(total_training_time)

    with open(os.path.join(REPORT_DIR, f"{model_tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    plot_history(history_head, history_fine, model_tag)

    print(f"\nResults for {model_tag}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    return metrics