"""
SonicSentinel AI - CNN Model for Audio Classification
Uses 2D Mel Spectrogram as input image with CNN architecture.
"""
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    SOUND_CLASSES, NUM_CLASSES, SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT,
    SEGMENT_DURATION, RANDOM_SEED, MODELS_DIR, MODEL_VERSION
)

# We use tensorflow/keras for CNN
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. CNN model will not be trained.")


TIME_FRAMES = int(np.ceil(SEGMENT_DURATION * SAMPLE_RATE / HOP_LENGTH)) + 1
CNN_INPUT_SHAPE = (N_MELS, TIME_FRAMES, 1)  # (128, ~130, 1)
CNN_MODEL_PATH = MODELS_DIR / "cnn_model.h5"


def build_cnn_model(
    input_shape: tuple = CNN_INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.4,
) -> "keras.Model":
    """
    Build CNN model for Mel spectrogram classification.
    Architecture:
      Conv2D (32) -> BN -> MaxPool -> Dropout
      Conv2D (64) -> BN -> MaxPool -> Dropout
      Conv2D (128) -> BN -> MaxPool -> Dropout
      Conv2D (256) -> BN -> GlobalAvgPool
      Dense (256) -> Dropout -> Dense (128) -> Dropout -> Softmax
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")

    tf.random.set_seed(RANDOM_SEED)

    inputs = keras.Input(shape=input_shape, name="mel_spectrogram")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="SonicSentinel_CNN")
    return model


def compile_cnn(model: "keras.Model", learning_rate: float = 1e-3) -> "keras.Model":
    """Compile CNN with Adam optimizer and label smoothing."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


def get_cnn_callbacks(
    checkpoint_path: str = str(CNN_MODEL_PATH),
    patience_lr: int = 5,
    patience_es: int = 15,
) -> list:
    """Return standard callbacks: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping."""
    if not TF_AVAILABLE:
        return []

    return [
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir=str(MODELS_DIR / "tensorboard_logs"),
            histogram_freq=1,
        ),
    ]


def train_cnn(
    X_train: np.ndarray,   # shape: (N, n_mels, time_frames)
    y_train_oh: np.ndarray, # one-hot encoded
    X_val: np.ndarray,
    y_val_oh: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
) -> Tuple["keras.Model", "keras.callbacks.History"]:
    """
    Train the CNN model on Mel spectrogram data.
    
    Args:
        X_train: Training Mel spectrograms (N, n_mels, time_frames)
        y_train_oh: One-hot encoded training labels
        X_val: Validation Mel spectrograms
        y_val_oh: One-hot encoded validation labels
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available.")

    tf.random.set_seed(RANDOM_SEED)

    # Add channel dimension
    X_train = X_train[..., np.newaxis]  # (N, n_mels, time, 1)
    X_val = X_val[..., np.newaxis]

    # Normalize mel spectrograms to [0, 1]
    global_min = X_train.min()
    global_max = X_train.max()
    X_train = (X_train - global_min) / (global_max - global_min + 1e-9)
    X_val = (X_val - global_min) / (global_max - global_min + 1e-9)

    # Build and compile
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_cnn_model(input_shape=input_shape, num_classes=y_train_oh.shape[1])
    model = compile_cnn(model, learning_rate=learning_rate)
    model.summary()

    cbks = get_cnn_callbacks()

    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbks,
        class_weight=compute_class_weights(np.argmax(y_train_oh, axis=1)),
        shuffle=True,
        verbose=1,
    )

    return model, history


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights to handle imbalance."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def predict_cnn(
    model: "keras.Model",
    mel_spec: np.ndarray,
    class_names: list = SOUND_CLASSES,
    global_min: float = -80.0,
    global_max: float = 0.0,
) -> Dict[str, Any]:
    """
    Run inference with CNN model.
    
    Args:
        mel_spec: 2D Mel spectrogram (n_mels, time_frames)
        class_names: List of class names
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available.")

    # Preprocess
    spec = mel_spec.copy()
    spec = (spec - global_min) / (global_max - global_min + 1e-9)
    spec = np.clip(spec, 0, 1)

    # Add batch + channel dims
    spec = spec[np.newaxis, ..., np.newaxis]  # (1, n_mels, time, 1)

    proba = model.predict(spec, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = class_names[pred_idx]
    top_confidence = float(proba[pred_idx])

    confidence_scores = {cls: float(proba[i]) for i, cls in enumerate(class_names)}
    sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
    top_two_margin = sorted_scores[0][1] - sorted_scores[1][1] if len(sorted_scores) > 1 else 1.0

    return {
        "predicted_class": pred_class,
        "confidence": top_confidence,
        "confidence_scores": confidence_scores,
        "top_two_margin": float(top_two_margin),
        "top_n_predictions": sorted_scores[:3],
        "model_version": MODEL_VERSION,
    }
