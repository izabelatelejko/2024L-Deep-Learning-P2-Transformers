"""Module for data preprocessing and augmentation."""

import random

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

import librosa

from tqdm import tqdm

from src.const import AUDIO_PATH, MAIN_LABELS, BATCH_SIZE, VALIDATION_SPLIT, SEED


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


def save_data(audio, labels):
    np.save("data.npy", audio)
    np.save("labels.npy", labels)


def load_augmented_data():
    return np.load("augmented_data.npy"), np.load("augmented_labels.npy")


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
    np.random.seed(SEED)
    random.seed(SEED)
    return mask_freqs(mask_time(ds))


def dataset_to_np(dataset):
    dataset = dataset.shuffle(100)
    dataset_len = tf.data.experimental.cardinality(dataset).numpy()
    X_filtered = [None for i in range(dataset_len)]
    y_filtered = [None for i in range(dataset_len)]

    i = 0
    for X, y in tqdm(dataset, "Processing dataset"):
        X_filtered[i] = X.numpy()
        y_filtered[i] = y.numpy()
        i += 1

    return np.array(X_filtered), np.array(y_filtered)


def normalize_ds(ds):
    for i in tqdm(range(ds.shape[0]), f"Normalizing dataset"):
        ds[i] = librosa.util.normalize(ds[i])

    return ds
