"""
Projet simple: Active Learning par entropie sur caractères arabes manuscrits
"""

from datetime import datetime

from src.data_loader import load_arabic_dataset
from src.active_learner import ActiveLearner

def main():
    print("="*80)
    print(" ACTIVE LEARNING - MANUSCRIT ARABE (SIMPLE ENTROPY)")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 1) Charger les données
    X_train, X_test, y_train, y_test, class_names = load_arabic_dataset()
    print(f"\n Données chargées: {len(class_names)} classes")

    # 2) Créer l'Active Learner (stratégie entropie + Random Forest)
    learner = ActiveLearner(X_train, y_train, X_test, y_test)

    # 3) Lancer la boucle d'Active Learning
    history = learner.run()

    # 4) Afficher résultat final
    print("\nRésumé:")
    print(f" - Accuracy finale: {history['accuracy'][-1]:.4f}")
    print(f" - F1-macro final: {history['f1_macro'][-1]:.4f}")
    print(f" - Nombre total d'exemples annotés: {history['n_labeled'][-1]}")

if __name__ == "__main__":
    main()
