import keras
from keras import layers
from keras import Model

from quantum import QuantumLayer
from classical import load_feature_extractor
import src.configs as cfg

def build_hybrid_model(model_path, n_qubits, q_depth=2, freeze_backbone=True):
    feature_extractor = load_feature_extractor(model_path)
    feature_extractor.trainable = not freeze_backbone

    inputs = keras.Input(shape=(cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3), name="input_image")
    x = feature_extractor(inputs, training=False)

    # Compress classical features to match qubit count
    x = layers.Dense(64, activation="relu", name="pre_quantum_dense")(x)
    x = layers.Dropout(0.2, name="pre_quantum_dropout")(x)
    x = layers.Dense(n_qubits, activation="tanh", name="quantum_input_projection")(x)

    x = QuantumLayer(n_qubits=n_qubits, q_depth=q_depth, name=f"quantum_{n_qubits}q")(x)

    x = layers.Dense(16, activation="relu", name="post_quantum_dense")(x)
    x = layers.Dropout(0.2, name="post_quantum_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="hybrid_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name=f"hybrid_{n_qubits}q")
    return model, feature_extractor

def unfreeze_top_fraction(feature_extractor, fraction=0.30):
    feature_extractor.trainable = True
    fine_tune_at = int(len(feature_extractor.layers) * (1 - fraction))

    for layer in feature_extractor.layers[:fine_tune_at]:
        layer.trainable = False

    for layer in feature_extractor.layers[fine_tune_at:]:
        layer.trainable = True

    # Keep normalization layers frozen for more stable fine-tuning
    for layer in feature_extractor.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False