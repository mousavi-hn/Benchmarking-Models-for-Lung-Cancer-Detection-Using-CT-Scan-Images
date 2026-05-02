import os
import time
import json

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from data.loader import make_generators
from data.dataset import OUTPUT_DIR, MODEL_DIR
from models.build_models import build_transfer_model, MODEL_CONFIGS
from evaluate.metrics import calculate_metrics
from evaluate.plots import plot_history
from src.configs import IMG_SIZE, BATCH_SIZE, LEARNING_RATE_HEAD, LEARNING_RATE_FINE, EPOCHS_HEAD,  EPOCHS_FINE

# TRAIN AND EVALUATE ONE MODEL
def train_and_evaluate(model_name, train_df, val_df, test_df):
    print(f"\n{'='*60}")
    print(f"Training model: {model_name}")
    print(f"{'='*60}")

    preprocess_func = MODEL_CONFIGS[model_name]["preprocess"]

    train_gen, val_gen, test_gen = make_generators(
        preprocess_func=preprocess_func,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    model, base_model = build_transfer_model(
        model_name=model_name,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # Stage 1: train classifier head
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model_path_best = os.path.join(MODEL_DIR, f"{model_name}_best.keras")

    callbacks_head = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1),
        ModelCheckpoint(model_path_best, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    start_time = time.time()

    history_head = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks_head,
        verbose=1
    )

    # Stage 2: fine-tune top layers
    base_model.trainable = True

    # Freeze lower layers, unfreeze upper layers only
    fine_tune_at = int(len(base_model.layers) * 0.7)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_fine = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1),
        ModelCheckpoint(model_path_best, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINE,
        callbacks=callbacks_fine,
        verbose=1
    )

    total_training_time = time.time() - start_time

    # Load best saved version
    best_model = tf.keras.models.load_model(model_path_best)

    # Evaluate
    y_true = test_gen.classes
    y_prob = best_model.predict(test_gen).ravel()

    metrics = calculate_metrics(y_true, y_prob, threshold=0.5)
    metrics["model_name"] = model_name
    metrics["training_time_sec"] = total_training_time

    plot_history(history_head, history_fine, model_name)

    # Save metrics JSON
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nResults for {model_name}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    return metrics