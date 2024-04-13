"""Module for loading the data."""

import os

import tensorflow as tf

from src.const import AUDIO_PATH, MAIN_LABELS, BATCH_SIZE, VALIDATION_SPLIT, SEED


def load_data():
    """Load the data."""
    labels_list = []

    for class_dir in os.listdir(AUDIO_PATH):
        labels_list += [
            MAIN_LABELS.index(class_dir) if class_dir in MAIN_LABELS else 10
            for _ in os.listdir(os.path.join(AUDIO_PATH, class_dir))
        ]

    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=AUDIO_PATH,
        labels=labels_list,
        batch_size=None,
        validation_split=VALIDATION_SPLIT,
        seed=SEED,
        output_sequence_length=16000,
        subset="both",
    )
    return train_ds, val_ds
