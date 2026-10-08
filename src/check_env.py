import sys
import numpy as np
import scipy
import pandas as pd
import librosa
import soundfile as sf
import sklearn
import matplotlib
import seaborn as sns
import tensorflow as tf
import tqdm

try:
    import jupyter_core
    jupyter_version = jupyter_core.__version__
except ImportError:
    import IPython
    jupyter_version = IPython.__version__

def main():
    print("=" * 60)
    print("ENVIRONMENT & DEPENDENCY VERSIONS")
    print("=" * 60)
    print(f"Python:       {sys.version.split()[0]}")
    print(f"numpy:        {np.__version__}")
    print(f"scipy:        {scipy.__version__}")
    print(f"pandas:       {pd.__version__}")
    print(f"librosa:      {librosa.__version__}")
    print(f"soundfile:    {sf.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"matplotlib:   {matplotlib.__version__}")
    print(f"seaborn:      {sns.__version__}")
    print(f"tensorflow:   {tf.__version__}")
    print(f"jupyter-core: {jupyter_version}")
    print(f"tqdm:         {tqdm.__version__}")
    print("=" * 60)

    # 1. Generate 1 second of synthetic audio (sine wave, 16 kHz, 440 Hz)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 440.0
    audio_signal = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    print(f"Generated synthetic audio: {len(audio_signal)} samples ({sr} Hz, {duration}s)")

    # 2. Extract 30 MFCCs with librosa and print output shape
    n_mfcc = 30
    mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=n_mfcc)
    print(f"Extracted {n_mfcc} MFCCs - Shape: {mfccs.shape} (n_mfcc, time_frames)")

    # 3. Build tiny Keras Conv1D model on random data and run one training step
    # Input shape: (batch_size, steps, channels)
    # Using tiny Conv1D architecture with Adam optimizer
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 1)),
        tf.keras.layers.Conv1D(filters=8, kernel_size=3, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=2, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    dummy_x = np.random.randn(8, 32, 1).astype(np.float32)
    dummy_y = np.random.randint(0, 2, size=(8,)).astype(np.int32)

    loss, acc = model.train_on_batch(dummy_x, dummy_y)
    print(f"Tiny Conv1D model single training step completed - Loss: {loss:.4f}, Acc: {acc:.4f}")

    print("=" * 60)
    print("ENVIRONMENT OK")
    print("=" * 60)

if __name__ == "__main__":
    main()
