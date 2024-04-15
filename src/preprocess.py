"""Module for data preprocessing and augmentation."""

import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from src.augmenter import generate_augmenter
from src.const import AUDIO_PATH, MAIN_LABELS, BATCH_SIZE, VALIDATION_SPLIT, SEED
from src.data_loader import load_data
from src.visualization import plot_augmented_samples, plot_spectograms


@tf.autograph.experimental.do_not_convert
def convert_to_array(ds):
    return np.asarray(list(ds.map(lambda x, y: x))), np.asarray(
        list(ds.map(lambda x, y: y))
    )


def extract_main_data(audio, labels):
    return audio[labels != 10], labels[labels != 10]


def augment(audio_array, augmenter):
    np.random.seed(SEED)
    random.seed(SEED)
    return np.apply_along_axis(augmenter, -1, audio_array, sample_rate=16000)


def combine_with_augmented(audio, audio_aug, labels, labels_aug):
    audio_with_aug = np.concatenate([audio, audio_aug], axis=0)
    labels_with_aug = np.concatenate([labels, labels_aug], axis=0)
    return audio_with_aug, labels_with_aug


def save_augmented_data(audio, labels):
    np.save("augmented_data.npy", audio)
    np.save("augmented_labels.npy", labels)


def load_augmented_data():
    return np.load("augmented_data.npy"), np.load("augmented_labels.npy")


def preprocess_and_save(plot_samples: bool = False):

    # Load data
    train_ds, _ = load_data()

    # Convert to numpy arrays
    train_audio, train_labels = convert_to_array(train_ds)
    del train_ds

    # Extract data with main labels
    train_audio_main, train_labels_main = extract_main_data(train_audio, train_labels)
    del train_audio

    # Augment data
    augmenter = generate_augmenter()
    train_audio_main_augmented = augment(train_audio_main, augmenter)

    # Plot augmented samples
    if plot_samples:
        plot_augmented_samples(train_audio_main[0:9], train_audio_main_augmented[0:9])
    del augmenter

    # Combine original and augmented data
    train_audio_with_augmented, train_labels_with_augmented = combine_with_augmented(
        train_audio_main,
        train_audio_main_augmented,
        train_labels,
        train_labels_main,
    )
    del train_audio_main, train_audio_main_augmented, train_labels, train_labels_main

    # Save original and augmented data
    save_augmented_data(train_audio_with_augmented, train_labels_with_augmented)
    print("Successfully saved augmented data.")
    del train_audio_with_augmented, train_labels_with_augmented


def create_binary_labels(labels):
    """Create binary labels for known vs unknown task. Known classes are labeled as 0, unknown as 1."""
    binary_labels = np.zeros(labels.shape)
    binary_labels[labels != 10] = 0
    binary_labels[labels == 10] = 1
    return binary_labels


def transform_to_spectograms(
    ds,
    nfft=512,
    window=512,
    stride=256,
    rate=16000,
    mels=128,
    fmin=0,
    fmax=8000,
    top_db=80,
):

    spect_ds = ds.map(
        map_func=lambda audio, label: (
            tfio.audio.spectrogram(audio, nfft=nfft, window=window, stride=stride),
            label,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    mel_spect_ds = spect_ds.map(
        map_func=lambda audio, label: (
            tfio.audio.melscale(audio, rate=rate, mels=mels, fmin=fmin, fmax=fmax),
            label,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return mel_spect_ds.map(
        map_func=lambda audio, label: (tfio.audio.dbscale(audio, top_db=top_db), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def map_val_labels(label):
    return tf.where(label < 10, 0, 1)


def mask_freqs(ds):
    return ds.map(
        map_func=lambda audio, label: (tfio.audio.freq_mask(audio, param=5), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def mask_time(ds):
    return ds.map(
        map_func=lambda audio, label: (tfio.audio.time_mask(audio, param=5), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def augment_spectograms(ds):
    return mask_freqs(mask_time(ds))


def load_and_preprocess_for_binary_task(plot_samples: bool = False):
    print(1)
    # Load augmented data
    train_audio_with_augmented, train_labels_with_augmented = load_augmented_data()

    print(2)
    # Create binary labels
    train_ds_binary = tf.data.Dataset.from_tensor_slices(
        (train_audio_with_augmented, create_binary_labels(train_labels_with_augmented))
    )
    del train_audio_with_augmented, train_labels_with_augmented

    print(2.5)
    _, val_ds = load_data()
    val_ds_binary = val_ds.map(lambda x, y: (x, map_val_labels(y)))
    del val_ds

    print(3)
    # Transform to spectograms
    train_ds_specs = transform_to_spectograms(train_ds_binary)
    val_ds_specs = transform_to_spectograms(val_ds_binary)
    del train_ds_binary, val_ds_binary
    print(4)
    # Augment spectograms
    train_ds_specs = augment_spectograms(train_ds_specs)
    print(5)
    if plot_samples:
        plot_spectograms(train_ds_specs)

    return train_ds_specs, val_ds_specs
