"""Central configuration module for the Speech Emotion Recognition (SER) project.

Holds audio preprocessing hyperparameters, random seeds, directory paths,
and dataset label mappings so all scripts share identical constants.
"""

from pathlib import Path

# ==============================================================================
# Shared Audio & Processing Settings
# ==============================================================================
SAMPLE_RATE = 16000   # Standard sampling frequency in Hz (16 kHz)
N_FFT = 512           # Number of FFT points for frequency analysis
HOP_LENGTH = 256      # Number of audio samples between successive STFT frames
RANDOM_SEED = 42      # Seed for reproducibility across random splits and initializations

# ==============================================================================
# Folder Paths
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_RAW_EMODB_DIR = DATA_RAW_DIR / "emodb"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure runtime directories exist
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# EmoDB Dataset Mappings
# ==============================================================================
# Emotion single-letter code in filename (character at index 5) -> emotion name
EMOTION_CODE_MAP = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "N": "neutral",
    "T": "sadness",
}

# Emotion labels assigned alphabetically (0 to 6)
# anger (0), boredom (1), disgust (2), fear (3), happiness (4), neutral (5), sadness (6)
EMOTION_LABEL_MAP = {
    "anger": 0,
    "boredom": 1,
    "disgust": 2,
    "fear": 3,
    "happiness": 4,
    "neutral": 5,
    "sadness": 6,
}

# Reverse mapping: integer label -> emotion name
LABEL_EMOTION_MAP = {v: k for k, v in EMOTION_LABEL_MAP.items()}

# Speaker ID (characters 1-2 in filename) -> Gender ('m' or 'f')
SPEAKER_GENDER_MAP = {
    "03": "m",
    "08": "f",
    "09": "f",
    "10": "m",
    "11": "m",
    "12": "m",
    "13": "f",
    "14": "f",
    "15": "m",
    "16": "f",
}
