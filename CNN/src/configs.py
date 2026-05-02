import tensorflow as tf

SEED = 42

IMG_SIZE = (224, 224)   # For fairness, I have used same input size for all models.
BATCH_SIZE = 32
EPOCHS_HEAD = 8         # training classifier head first
EPOCHS_FINE = 7         # then fine-tuning upper layers
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 1e-5
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

#  MODEL LIST
MODEL_CONFIGS = {
    "VGG16": {
        "builder": tf.keras.applications.VGG16,
        "preprocess": tf.keras.applications.vgg16.preprocess_input
    },
    "VGG19": {
        "builder": tf.keras.applications.VGG19,
        "preprocess": tf.keras.applications.vgg19.preprocess_input
    },
    "ResNet50V2": {
        "builder": tf.keras.applications.ResNet50V2,
        "preprocess": tf.keras.applications.resnet_v2.preprocess_input
    },
    "DenseNet121": {
        "builder": tf.keras.applications.DenseNet121,
        "preprocess": tf.keras.applications.densenet.preprocess_input
    },
    "DenseNet201": {
        "builder": tf.keras.applications.DenseNet201,
        "preprocess": tf.keras.applications.densenet.preprocess_input
    },
    "EfficientNetB0": {
        "builder": tf.keras.applications.EfficientNetB0,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input
    },
    "MobileNetV2": {
        "builder": tf.keras.applications.MobileNetV2,
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input
    },
    "InceptionV3": {
        "builder": tf.keras.applications.InceptionV3,
        "preprocess": tf.keras.applications.inception_v3.preprocess_input
    },
    "Xception": {
        "builder": tf.keras.applications.Xception,
        "preprocess": tf.keras.applications.xception.preprocess_input
    }
}

MODEL_NAMES = list(MODEL_CONFIGS.keys())