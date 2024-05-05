"""Module for visualization of data."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_single_augmented(non_augmented, augmented, ax):
    ax.plot(np.arange(0, len(non_augmented)), augmented, color="pink")
    ax.plot(np.arange(0, len(non_augmented)), non_augmented, alpha=0.3, color="blue")


def plot_augmented_samples(non_augmented, augmented):
    fig, ax = plt.subplots(3, 3, figsize=(18, 16))
    for i in range(9):
        plot_single_augmented(non_augmented[i], augmented[i], ax[i // 3, i % 3])
    plt.show()


def plot_spectograms(ds, n_samples=3):
    for audio, _ in ds.take(n_samples):
        plt.imshow(audio.numpy())
        plt.show()


def plot_conf_matrix(y_test, y_pred, classes):
    plt.figure(figsize=(8, 6))
    result = confusion_matrix(y_test, y_pred, normalize="true")
    sns.heatmap(result, annot=True, fmt=".2f", xticklabels=classes, yticklabels=classes)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion matrix on test data")
    plt.show()


def plot_losses(train_losses, val_losses, title):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["Train loss", "Val loss"])
    plt.xlabel("Epoch")
    plt.title(title)
    plt.show()
