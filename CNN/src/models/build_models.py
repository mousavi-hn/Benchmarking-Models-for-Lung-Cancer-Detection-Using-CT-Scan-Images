from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input

from src.configs import MODEL_CONFIGS

# BUILD MODEL
def build_transfer_model(model_name, input_shape=(224, 224, 3), dropout_rate=0.3):
    config = MODEL_CONFIGS[model_name]
    base_builder = config["builder"]

    base_model = base_builder(
        include_top=False,
        weights="imagenet",
        input_tensor=Input(shape=input_shape)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model