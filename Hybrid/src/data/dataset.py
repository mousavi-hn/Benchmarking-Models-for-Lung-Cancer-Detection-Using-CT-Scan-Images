import keras
import numpy as np

from loader import read_image
import src.configs as cfg

class MRISequence(keras.utils.Sequence):
    def __init__(self, df, preprocess_func, batch_size=8, target_size=(224, 224), shuffle=False):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.preprocess_func = preprocess_func
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        images = []
        labels = []

        for _, row in batch_df.iterrows():
            img = read_image(row["filepath"], self.target_size)
            img = self.preprocess_func(img)
            images.append(img)
            labels.append(float(row["label"]))

        x = np.asarray(images, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            rng = np.random.default_rng(cfg.SEED)
            rng.shuffle(self.indices)

def make_generators(preprocess_func, train_df, val_df, test_df):
    train_seq = MRISequence(
        train_df, preprocess_func, batch_size=cfg.BATCH_SIZE,
        target_size=cfg.IMG_SIZE, shuffle=True
    )
    val_seq = MRISequence(
        val_df, preprocess_func, batch_size=cfg.BATCH_SIZE,
        target_size=cfg.IMG_SIZE, shuffle=False
    )
    test_seq = MRISequence(
        test_df, preprocess_func, batch_size=cfg.BATCH_SIZE,
        target_size=cfg.IMG_SIZE, shuffle=False
    )
    return train_seq, val_seq, test_seq