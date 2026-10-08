from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "emodb"
PROC_DIR = ROOT / "data" / "processed"
FEAT_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"

SR = 16000
N_FFT = 512
HOP_LENGTH = 256
SEED = 42
COEFF_LIST = [12, 24, 30]
N_ROWS = 535

EMOTION_MAP = {"W": "anger", "L": "boredom", "E": "disgust", "A": "fear",
               "F": "happiness", "T": "sadness", "N": "neutral"}
SPEAKER_GENDER = {"03": "m", "08": "f", "09": "f", "10": "m", "11": "m",
                  "12": "m", "13": "f", "14": "f", "15": "m", "16": "f"}
