"""
Boucle principale d'Active Learning (stratégie: entropy)
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import sys

# Ajouter la racine du projet au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import INITIAL_LABELED_SIZE, QUERY_BATCH_SIZE, N_ITERATIONS
from config import STRATEGY_NAME, MODEL_TYPE
from src.models import ModelWrapper
from src.query_strategies import get_query_strategy


class ActiveLearner:
    def __init__(self, X_train, y_train, X_test, y_test,
                 strategy_name=STRATEGY_NAME, model_type=MODEL_TYPE):

        self.X_pool = X_train.copy()
        self.y_pool = y_train.copy()
        self.X_test = X_test
        self.y_test = y_test

        self.strategy_name = strategy_name
        self.strategy = get_query_strategy(strategy_name)
        self.model = ModelWrapper(model_type)

        self.X_labeled = None
        self.y_labeled = None
        self.labeled_indices = []
        self.unlabeled_indices = list(range(len(X_train)))

        self.history = {
            "n_labeled": [],
            "accuracy": [],
            "f1_macro": [],
            "f1_micro": []
        }

        print("=" * 60)
        print(f" Active Learner | stratégie={self.strategy_name}, modèle={model_type}")
        print(f" Pool: {len(self.X_pool)} | Test: {len(self.X_test)}")
        print("=" * 60)

    def initialize_labeled_set(self, n_initial=INITIAL_LABELED_SIZE):
        print(f"\n Init avec {n_initial} exemples étiquetés aléatoirement...")
        initial_indices = np.random.choice(
            self.unlabeled_indices,
            size=n_initial,
            replace=False
        )
        self.X_labeled = self.X_pool[initial_indices]
        self.y_labeled = self.y_pool[initial_indices]
        self.labeled_indices = list(initial_indices)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in initial_indices]
        print(f" Labeled: {len(self.labeled_indices)} | Unlabeled: {len(self.unlabeled_indices)}")

    def train_model(self):
        self.model.fit(self.X_labeled, self.y_labeled)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1_macro = f1_score(self.y_test, y_pred, average="macro")
        f1_micro = f1_score(self.y_test, y_pred, average="micro")
        return acc, f1_macro, f1_micro

    def query_instances(self, n_instances=QUERY_BATCH_SIZE):
        X_unlabeled = self.X_pool[self.unlabeled_indices]
        local_idx = self.strategy.select(
            self.model.model,
            X_unlabeled,
            min(n_instances, len(self.unlabeled_indices))
        )
        selected = [self.unlabeled_indices[i] for i in local_idx]
        return selected

    def label_instances(self, indices):
        new_X = self.X_pool[indices]
        new_y = self.y_pool[indices]
        self.X_labeled = np.vstack([self.X_labeled, new_X])
        self.y_labeled = np.concatenate([self.y_labeled, new_y])
        self.labeled_indices.extend(indices)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in indices]

    def run(self, n_iterations=N_ITERATIONS, n_instances_per_iteration=QUERY_BATCH_SIZE):
        self.initialize_labeled_set()

        for it in range(n_iterations):
            if len(self.unlabeled_indices) == 0:
                print(f"\nPlus d'exemples non étiquetés (itération {it})")
                break

            print(f"\n--- Itération {it + 1} ---")
            print(f" Labeled: {len(self.labeled_indices)} | Unlabeled: {len(self.unlabeled_indices)}")

            self.train_model()
            acc, f1_macro, f1_micro = self.evaluate_model()

            self.history["n_labeled"].append(len(self.labeled_indices))
            self.history["accuracy"].append(acc)
            self.history["f1_macro"].append(f1_macro)
            self.history["f1_micro"].append(f1_micro)

            print(f" Accuracy={acc:.4f} | F1-macro={f1_macro:.4f}")

            selected = self.query_instances(n_instances_per_iteration)
            self.label_instances(selected)

        print("\n=== FIN ACTIVE LEARNING ===")
        print(f"Accuracy finale: {self.history['accuracy'][-1]:.4f}")
        print(f"Total exemples étiquetés: {self.history['n_labeled'][-1]}")
        return self.history
