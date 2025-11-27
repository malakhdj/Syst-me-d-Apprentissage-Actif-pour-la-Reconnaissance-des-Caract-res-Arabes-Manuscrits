"""
Stratégies de requête pour l'Active Learning
"""

import numpy as np
from scipy.stats import entropy
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class EntropySamplingQueryStrategy:
    def __init__(self):
        self.name = "entropy"

    def select(self, model, X_unlabeled, n_instances):
        """
        Sélectionne les n_instances exemples avec l'entropie maximale.
        """
        # Probabilités de prédiction
        probas = model.predict_proba(X_unlabeled)
        # Entropie pour chaque exemple
        entropies = entropy(probas.T)
        # Indices des plus grandes entropies
        indices = np.argsort(entropies)[-n_instances:]
        return indices

def get_query_strategy(strategy_name: str):
    if strategy_name != "entropy":
        raise ValueError("Dans ce projet simple, seule la stratégie 'entropy' est supportée.")
    return EntropySamplingQueryStrategy()
