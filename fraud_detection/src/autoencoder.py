import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def build_autoencoder(input_dim=30):
    # I use vim btw :D
    inputs = keras.Input(shape=(input_dim,))
    # Encoder
    x = keras.layers.Dense(24, activation="relu")(inputs)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    encoded = keras.layers.Dense(4, activation="relu")(x)
    # Decoder
    x = keras.layers.Dense(8, activation="relu")(encoded)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dense(24, activation="relu")(x)
    outputs = keras.layers.Dense(input_dim, activation="linear")(x)

    model = keras.Model(inputs, outputs, name="autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def train_autoencoder(X_ae_train, model_path):
    """
    Train autoencoder on normal transactions only (input = target).

    Returns:
        model   - trained keras Model
        history - training History object
    """
    tf.random.set_seed(42)

    model = build_autoencoder(input_dim=X_ae_train.shape[1])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    history = model.fit(
        X_ae_train,
        X_ae_train,
        epochs=10_000,
        batch_size=256,
        validation_split=0.1,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )

    # ModelCheckpoint already saved the best model; this is a no-op if path matches
    print(f"  Autoencoder saved to: {model_path}")
    print(f"  Stopped at epoch {len(history.history['loss'])}")

    return model, history


def compute_reconstruction_errors(model, X):
    """
    Compute per-sample MSE reconstruction error.

    Returns:
        errors - np.ndarray of shape (n,)
    """
    X_pred = model.predict(X, batch_size=512, verbose=0)
    errors = np.square(X - X_pred)  # shape (n, 30) — per-feature squared error
    return errors


def load_autoencoder(model_path):
    """Load a saved autoencoder from disk."""
    return keras.models.load_model(model_path)
