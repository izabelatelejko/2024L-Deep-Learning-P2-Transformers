"""Module for visualization of data."""

import matplotlib.pyplot as plt
import numpy as np


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
