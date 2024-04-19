"""Module for data preprocessing and augmentation."""

import tensorflow as tf
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.augmenter import generate_augmenter
from src.data_loader import load_data
from src.preprocess_utils import (
    augment,
    augment_spectograms,
    combine_with_augmented,
    convert_to_array,
    create_binary_labels,
    extract_main_data,
    load_augmented_data,
    map_val_labels,
    save_augmented_data,
    transform_to_spectograms,
    dataset_to_np,
)
from src.visualization import plot_augmented_samples, plot_spectograms


def preprocess_and_save(plot_samples: bool = False):
    """Load audio data, augment it, and save the augmented data as numpy array."""

    # Load data
    train_ds, _ = load_data()

    # Convert to numpy arrays
    train_audio, train_labels = convert_to_array(train_ds)
    del train_ds

    # Extract data with main labels
    train_audio_main, train_labels_main = extract_main_data(train_audio, train_labels)

    # Augment data
    augmenter = generate_augmenter()
    train_audio_main_augmented = augment(train_audio_main, augmenter)

    # Plot augmented samples
    if plot_samples:
        plot_augmented_samples(train_audio_main[0:9], train_audio_main_augmented[0:9])
    del augmenter, train_audio_main

    # Combine original and augmented data
    train_audio_with_augmented, train_labels_with_augmented = combine_with_augmented(
        train_audio,
        train_audio_main_augmented,
        train_labels,
        train_labels_main,
    )
    del train_audio, train_audio_main_augmented, train_labels, train_labels_main

    # Save original and augmented data
    save_augmented_data(train_audio_with_augmented, train_labels_with_augmented)
    print("Successfully saved augmented data.")
    del train_audio_with_augmented, train_labels_with_augmented


def load_and_preprocess(plot_samples: bool = False, augment_specs: bool = True):
    """Load augmented data, create binary labels, transform data to spectograms, and perform augmentation on spectograms."""

    print("Loading augmented data...")
    train_audio_with_augmented, train_labels_with_augmented = load_augmented_data()
    _, val_ds = load_data()
    print("Finished")

    print("Creating data with only main classes...")
    train_audio_with_augmented_main = train_audio_with_augmented[
        train_labels_with_augmented != 10
    ]
    train_labels_with_augmented_main = train_labels_with_augmented[
        train_labels_with_augmented != 10
    ]
    print("Finished")

    print("Creating binary dataset...")
    train_ds_binary = tf.data.Dataset.from_tensor_slices(
        (train_audio_with_augmented, create_binary_labels(train_labels_with_augmented))
    )
    del train_audio_with_augmented, train_labels_with_augmented
    print("Finished")

    print("Creating main dataset...")
    train_ds_main = tf.data.Dataset.from_tensor_slices(
        (train_audio_with_augmented_main, train_labels_with_augmented_main)
    )
    del train_audio_with_augmented_main, train_labels_with_augmented_main

    _, val_ds = load_data()
    val_ds_main = val_ds
    val_ds_binary = val_ds.map(lambda x, y: (x, map_val_labels(y)))
    del val_ds
    print("Finished")

    print("Transforming audio data to spectograms...")
    train_ds_specs_binary = transform_to_spectograms(train_ds_binary)
    val_ds_specs_binary = transform_to_spectograms(val_ds_binary)

    train_ds_specs_main = transform_to_spectograms(train_ds_main)
    val_ds_specs_main = transform_to_spectograms(val_ds_main)
    del train_ds_main, train_ds_binary, val_ds_binary, val_ds_main
    print("Finished")

    if augment_specs:
        print("Augmenting spectograms...")
        train_ds_specs_binary = augment_spectograms(train_ds_specs_binary)
        train_ds_specs_main = augment_spectograms(train_ds_specs_main)
        print("Finished")

    print("Transforming data to numpy arrays...")
    train_ds_specs_binary_X, train_ds_specs_binary_y = dataset_to_np(
        train_ds_specs_binary
    )
    val_ds_specs_binary_X, val_ds_specs_binary_y = dataset_to_np(val_ds_specs_binary)
    del val_ds_specs_binary

    train_ds_specs_main_X, train_ds_specs_main_y = dataset_to_np(train_ds_specs_main)
    del train_ds_specs_main

    val_ds_specs_main_X, val_ds_specs_main_y = dataset_to_np(val_ds_specs_main)
    del val_ds_specs_main
    print("Finished")

    val_ds_specs_main_X = val_ds_specs_main_X[val_ds_specs_main_y != 10]
    val_ds_specs_main_y = val_ds_specs_main_y[val_ds_specs_main_y != 10]

    # if plot_samples:
    #     plot_spectograms(train_ds_specs_binary)

    return (
        train_ds_specs_binary_X,
        train_ds_specs_binary_y,
        val_ds_specs_binary_X,
        val_ds_specs_binary_y,
        train_ds_specs_main_X,
        train_ds_specs_main_y,
        val_ds_specs_main_X,
        val_ds_specs_main_y,
    )


def transform_to_data_loader(X, y, device):
    """Transform numpy arrays to PyTorch DataLoader."""
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)
    del X, y

    dataset = TensorDataset(X_t, y_t)
    del X_t, y_t

    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    del dataset

    return data_loader
