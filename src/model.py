"""
1D-CNN Model Architecture for Speech Emotion Recognition (SER).

Reproducing the research paper:
"Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition"
Automatika (2024).

Model Architecture Specification (from paper & spec sheet):
  - Input: 1D acoustic feature vector of shape (input_dim, 1)
  - 4 Conv1D blocks, each with:
      * Conv1D with kernel_size=15, stride=1, padding='same'
      * BatchNormalization
      * ReLU activation
      * MaxPooling1D with pool_size=2
    Filter sizes:
      * Layer 1: 8 filters
      * Layer 2: 16 filters
      * Layer 3: 32 filters
      * Layer 4: 64 filters (or 62 filters per paper verbatim typo)
  - Dropout(0.3) after 4th conv block
  - Flatten
  - Dense(120, activation='relu')
  - Dropout(0.2)
  - Dense(num_classes, activation='softmax') (7 emotion classes for EmoDB)
  - Optimizer: Adam (learning_rate=0.001)
  - Loss: sparse_categorical_crossentropy
"""

import os
import sys
from pathlib import Path
import numpy as np

# Ensure project root and src directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Suppress verbose TensorFlow warnings where possible
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
import keras
from keras import layers


def build_1d_cnn(
    input_dim=60,
    input_shape=None,
    num_classes=7,
    filters=(8, 16, 32, 64),
    kernel_size=15,
    dense_units=120,
    dropout_conv=0.3,
    dropout_dense=0.2,
    learning_rate=0.001,
    name="ser_1d_cnn",
):
    """
    Build and compile the 1D-CNN model described in the paper.

    Parameters
    ----------
    input_dim : int
        Input feature vector dimension (e.g., 24, 48, 60 for 12, 24, 30 coeffs).
        Ignored if input_shape is explicitly provided.
    input_shape : tuple, optional
        Input shape tuple, e.g. (input_dim, 1) for 1D features or (144, n_coeff) for frame sequences.
    num_classes : int
        Number of output emotion classes (default 7 for EmoDB).
    filters : tuple of 4 ints
        Number of filters for the 4 Conv1D layers (default (8, 16, 32, 64)).
    kernel_size : int
        Convolution kernel size (default 15).
    dense_units : int
        Number of units in the penultimate Dense layer (default 120).
    dropout_conv : float
        Dropout rate applied after the 4th conv block (default 0.3).
    dropout_dense : float
        Dropout rate applied after the dense layer (default 0.2).
    learning_rate : float
        Learning rate for the Adam optimizer (default 0.001).
    name : str
        Model name.

    Returns
    -------
    keras.Model
        Compiled Keras 1D-CNN model.
    """
    if len(filters) != 4:
        raise ValueError(f"Expected 4 filter values for 4 Conv1D layers, got {len(filters)}: {filters}")

    if input_shape is None:
        input_shape = (input_dim, 1)

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape, name="input_features"),
            # Block 1
            layers.Conv1D(
                filters=filters[0],
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                name="conv1d_block1",
            ),
            layers.BatchNormalization(name="bn_block1"),
            layers.Activation("relu", name="relu_block1"),
            layers.MaxPooling1D(pool_size=2, name="pool_block1"),
            # Block 2
            layers.Conv1D(
                filters=filters[1],
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                name="conv1d_block2",
            ),
            layers.BatchNormalization(name="bn_block2"),
            layers.Activation("relu", name="relu_block2"),
            layers.MaxPooling1D(pool_size=2, name="pool_block2"),
            # Block 3
            layers.Conv1D(
                filters=filters[2],
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                name="conv1d_block3",
            ),
            layers.BatchNormalization(name="bn_block3"),
            layers.Activation("relu", name="relu_block3"),
            layers.MaxPooling1D(pool_size=2, name="pool_block3"),
            # Block 4
            layers.Conv1D(
                filters=filters[3],
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                name="conv1d_block4",
            ),
            layers.BatchNormalization(name="bn_block4"),
            layers.Activation("relu", name="relu_block4"),
            layers.MaxPooling1D(pool_size=2, name="pool_block4"),
            # Classification head
            layers.Dropout(dropout_conv, name="dropout_conv"),
            layers.Flatten(name="flatten"),
            layers.Dense(dense_units, activation="relu", name="dense_penultimate"),
            layers.Dropout(dropout_dense, name="dropout_dense"),
            layers.Dense(num_classes, activation="softmax", name="emotion_output"),
        ],
        name=name,
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def prepare_input(X):
    """
    Ensure input feature array has 3D shape (batch_size, input_dim, 1) for Conv1D.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (N, D) or 3D array of shape (N, D, 1).

    Returns
    -------
    np.ndarray
        3D float32 array with shape (N, D, 1).
    """
    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim == 2:
        X_arr = np.expand_dims(X_arr, axis=-1)
    elif X_arr.ndim != 3 or X_arr.shape[-1] != 1:
        raise ValueError(f"Expected shape (N, D) or (N, D, 1), got {X_arr.shape}")
    return X_arr


if __name__ == "__main__":
    print("===== 1D-CNN Model Architecture Summary (Input Dim = 60) =====")
    sample_model = build_1d_cnn(input_dim=60, filters=(8, 16, 32, 64))
    sample_model.summary()

    # Smoke test forward pass
    dummy_x = np.random.randn(16, 60, 1).astype(np.float32)
    preds = sample_model.predict(dummy_x, verbose=0)
    print(f"\nSmoke test passed: input shape {dummy_x.shape} -> predictions shape {preds.shape}")
    print(f"Total parameters: {sample_model.count_params():,}")
