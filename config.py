import os
import numpy as np

# Graine aléatoire
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Dossiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Chemin du dataset
DATASET_PATH = os.path.join(RAW_DATA_DIR, "dataset.txt")

# Paramètres du dataset
# >>> on utilise plus de features (16 premières colonnes numériques)
N_FEATURES = 64   # au lieu de 16
TEST_SIZE = 0.2

# Paramètres Active Learning
INITIAL_LABELED_SIZE = 40
QUERY_BATCH_SIZE = 20
N_ITERATIONS = 40   # plus d'itérations => plus d'exemples annotés

# Modèle (Random Forest un peu plus puissant)
# Modèle (SVM RBF, plus puissant sur features normalisées)
MODEL_TYPE = "svm"
MODEL_PARAMS = {
    "svm": {
        "kernel": "rbf",
        "C": 5.0,
        "gamma": "scale",
        "probability": True,
        "random_state": RANDOM_SEED
    }
}


# Stratégie (unique)
STRATEGY_NAME = "entropy"
