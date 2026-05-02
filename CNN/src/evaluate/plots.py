import os

import matplotlib.pyplot as plt

from src.data.dataset import PLOT_DIR

def plot_history(history_head, history_fine, model_name):
    acc = history_head.history.get("accuracy", []) + history_fine.history.get("accuracy", [])
    val_acc = history_head.history.get("val_accuracy", []) + history_fine.history.get("val_accuracy", [])
    loss = history_head.history.get("loss", []) + history_fine.history.get("loss", [])
    val_loss = history_head.history.get("val_loss", []) + history_fine.history.get("val_loss", [])

    plt.figure(figsize=(8, 5))
    plt.plot(acc, label="train_accuracy")
    plt.plot(val_acc, label="val_accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_accuracy.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_loss.png"))
    plt.close()