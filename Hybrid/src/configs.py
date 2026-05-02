import keras

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

EPOCHS_HEAD = 8
EPOCHS_FINE = 5

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 1e-5

QUANTUM_QUBITS = [2, 4, 6, 8, 12, 16]
Q_DEPTH = 2

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

MODEL_CONFIGS = {
    "VGG16": {
        "preprocess": keras.applications.vgg16.preprocess_input
    },
    "VGG19": {
        "preprocess": keras.applications.vgg19.preprocess_input
    },
    "ResNet50V2": {
        "preprocess": keras.applications.resnet_v2.preprocess_input
    },
    "DenseNet121": {
        "preprocess": keras.applications.densenet.preprocess_input
    },
    "DenseNet201": {
        "preprocess": keras.applications.densenet.preprocess_input
    },
    "EfficientNetB0": {
        "preprocess": keras.applications.efficientnet.preprocess_input
    },
    "MobileNetV2": {
        "preprocess": keras.applications.mobilenet_v2.preprocess_input
    },
    "InceptionV3": {
        "preprocess": keras.applications.inception_v3.preprocess_input
    },
    "Xception": {
        "preprocess": keras.applications.xception.preprocess_input
    }
}

MODEL_NAMES = list(MODEL_CONFIGS.keys())