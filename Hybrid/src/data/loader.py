from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.configs import SEED

def make_generators(preprocess_func, train_df, val_df, test_df, target_size=(224, 224), batch_size=32):

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func
    )

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="file_path",
        y_col="label",
        target_size=target_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=True,
        seed=SEED
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="file_path",
        y_col="label",
        target_size=target_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False
    )

    test_gen = datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="file_path",
        y_col="label",
        target_size=target_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False
    )

    return train_gen, val_gen, test_gen