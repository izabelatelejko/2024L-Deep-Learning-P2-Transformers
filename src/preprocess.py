"""Module for data preprocessing and augmentation."""

import random

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

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


def load_augmented_data():
    return np.load("augmented_data.npy"), np.load("augmented_labels.npy")


def preprocess_data_into_specs(
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
