# Fichier : scripts/2_train_models.py
# Script d'entraînement V6 SANS SMOTE (Correction Finale)
# ========================================================
#
# Ce script entraîne les 2 modèles ML du projet en séquence.
#
# CONTEXTE :
# Ce fichier est le cœur du projet. Il transforme les données prétraitées
# en modèles ML capables de prédire les coupures d'électricité.
#
# VERSION 6 - CORRECTION MAJEURE :
# Problème résolu : Inversion des prédictions (Dakar-Plateau prédit comme
# plus risqué que Guediawaye, alors que c'est l'inverse dans les données).
#
# Cause : SMOTE créait des données synthétiques qui mélangeaient les quartiers
# Solution : Suppression de SMOTE + scale_pos_weight=10.0 uniquement
#
# DURÉE : ~10 minutes (LightGBM 2 min, LSTM 8 min)

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Ajouter le dossier racine au path pour importer src/
sys.path.append(str(Path(__file__).parent.parent))

# Import des modules internes
try:
    from src.data_pipeline import DataPipeline
    from src.config import (
        LGBM_MODEL_FILE, LSTM_MODEL_FILE,
        SEQUENCE_LENGTH, LSTM_EPOCHS, LSTM_BATCH_SIZE
    )
except ImportError as e:
    print(f"⚠️ AVERTISSEMENT: Impossible d'importer un module interne. Erreur: {e}")
    LGBM_MODEL_FILE = Path("models/lgbm_model.joblib")
    LSTM_MODEL_FILE = Path("models/lstm_model.h5")
    SEQUENCE_LENGTH = 12
    LSTM_EPOCHS = 50

# Batch size par défaut
DEFAULT_LSTM_BATCH_SIZE = 256
if 'LSTM_BATCH_SIZE' not in locals():
    LSTM_BATCH_SIZE = DEFAULT_LSTM_BATCH_SIZE

# ML
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# DL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class ImprovedModelTrainer:
    """
    Entraîneur V6 - SANS SMOTE pour éviter l'inversion des corrélations.
    
    Cette classe gère l'entraînement complet des 2 modèles du projet.
    Elle orchestre : préparation données → entraînement → évaluation → sauvegarde.
    """
    
    def __init__(self):
        """Initialise l'entraîneur avec le pipeline de données."""
        if 'DataPipeline' in globals():
            self.pipeline = DataPipeline()
        else:
            raise RuntimeError("La classe DataPipeline n'a pas pu être chargée.")
        
        self.lgbm_model = None
        self.lstm_model = None
    
    def train_lgbm_improved(self, X_train, y_train, X_test, y_test):
        """
        Entraîne LightGBM SANS SMOTE (correction V6).
        
        AVANT V6 : SMOTE créait des données synthétiques qui inversaient les corrélations
        APRÈS V6 : Données réelles uniquement + scale_pos_weight=10.0
        
        Résultat : Guediawaye > Dakar-Plateau (cohérent avec les données)
        """
        print("\n" + "="*60)
        print("🌳 ENTRAÎNEMENT LIGHTGBM V6 (SANS SMOTE)")
        print("="*60)
        
        # Afficher la distribution (vérifier le déséquilibre 93%/7%)
        print(f"\n⚖️ Distribution originale (AUCUN rééchantillonnage) :")
        print(f"   Classe 0 (pas de coupure) : {(y_train == 0).sum():,}")
        print(f"   Classe 1 (coupure)        : {(y_train == 1).sum():,}")
        print(f"   Ratio coupures            : {y_train.mean()*100:.2f}%")
        
        # Paramètres LightGBM optimisés V6
        # Ces valeurs ont été ajustées pour éviter l'overfitting sur quartier_encoded
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,          # Réduit (était 40-60 en V5)
            'max_depth': 6,            # Réduit (était 8-10)
            'learning_rate': 0.05,     # Augmenté (était 0.02-0.03)
            'feature_fraction': 0.9,   # Augmenté (était 0.8)
            'bagging_fraction': 0.9,   # Augmenté
            'bagging_freq': 5,
            'min_child_samples': 50,   # Augmenté (était 20-30)
            'min_split_gain': 0.01,    # Nouveau en V6
            'reg_alpha': 0.1,          # Nouveau en V6 (régularisation L1)
            'reg_lambda': 0.1,         # Nouveau en V6 (régularisation L2)
            'scale_pos_weight': 10.0,  # Augmenté (était 3.3 avec SMOTE)
            'verbose': -1,
            'n_estimators': 500,
            'random_state': 42
        }
        
        # Créer les datasets LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Entraîner
        print("\n🔄 Entraînement en cours (données réelles uniquement)...")
        self.lgbm_model = lgb.train(
            params,
            train_data,
            num_boost_round=params['n_estimators'],
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        
        print("\n✅ Entraînement terminé !")
        
        # Importance des features (vérifier que quartier n'est pas trop dominant)
        print("\n📊 Importance des features (top 5) :")
        feature_importance = self.lgbm_model.feature_importance(importance_type='gain')
        feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        for idx, row in importance_df.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.0f}")
        
        # Trouver le seuil optimal
        print("\n📊 Recherche du seuil optimal...")
        y_pred_proba = self.lgbm_model.predict(X_test)
        
        best_threshold = self._find_best_threshold(y_test, y_pred_proba)
        print(f"   🎯 Seuil optimal trouvé : {best_threshold:.3f}")
        
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        # Afficher les métriques
        self._print_metrics(y_test, y_pred, y_pred_proba, "LightGBM V6 (Sans SMOTE)")
        
        # Sauvegarder le modèle + seuil
        LGBM_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.lgbm_model,
            'threshold': best_threshold
        }, LGBM_MODEL_FILE)
        print(f"\n💾 Modèle sauvegardé : {LGBM_MODEL_FILE}")
        
        return self.lgbm_model
    
    def build_improved_lstm(self, input_shape):
        """
        Construit l'architecture LSTM optimisée.
        
        Architecture :
        - LSTM 100 units (return_sequences=True)
        - BatchNorm + Dropout 40%
        - LSTM 50 units
        - BatchNorm + Dropout 40%
        - Dense 32 + BatchNorm + Dropout 30%
        - Dense 16 + Dropout 20%
        - Dense 1 (sigmoid)
        
        Total : ~77,000 paramètres
        """
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.4),
            
            LSTM(50, return_sequences=False),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def train_lstm_improved(self, X_train, y_train, X_test, y_test):
        """
        Entraîne le modèle LSTM.
        
        LSTM est moins performant que LightGBM sur ce dataset (52k lignes),
        mais il est utile pour comparaison et ensemble learning.
        """
        print("\n" + "="*60)
        print("🧠 ENTRAÎNEMENT LSTM V6")
        print("="*60)
        
        # Créer les séquences temporelles (12 heures d'historique)
        print(f"\n🔄 Création des séquences (longueur={SEQUENCE_LENGTH})...")
        X_train_seq, y_train_seq = self.pipeline.create_sequences(X_train, y_train, SEQUENCE_LENGTH)
        X_test_seq, y_test_seq = self.pipeline.create_sequences(X_test, y_test, SEQUENCE_LENGTH)
        
        print(f"   Train: {X_train_seq.shape}")
        print(f"   Test:  {X_test_seq.shape}")
        
        # Poids des classes (compenser le déséquilibre)
        neg_count = (y_train_seq == 0).sum()
        pos_count = (y_train_seq == 1).sum()
        class_weight = {0: 1.0, 1: neg_count / pos_count}
        print(f"\n⚖️ Poids des classes : {class_weight}")
        
        # Construire le modèle
        print("\n🏗️ Construction du modèle LSTM...")
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.lstm_model = self.build_improved_lstm(input_shape)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                str(LSTM_MODEL_FILE),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=10,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Entraîner
        print(f"\n🔄 Entraînement en cours (Batch: {LSTM_BATCH_SIZE}, Epochs: {LSTM_EPOCHS})...")
        history = self.lstm_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2
        )
        
        print("\n✅ Entraînement terminé !")
        
        # Évaluation
        print("\n📊 Évaluation sur le test set :")
        y_pred_proba = self.lstm_model.predict(X_test_seq, verbose=0).flatten()
        
        best_threshold = self._find_best_threshold(y_test_seq, y_pred_proba)
        print(f"   🎯 Seuil optimal trouvé : {best_threshold:.3f}")
        
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        self._print_metrics(y_test_seq, y_pred, y_pred_proba, "LSTM V6")
        
        # Sauvegarder le seuil
        threshold_file = LSTM_MODEL_FILE.parent / "lstm_threshold.txt"
        with open(threshold_file, 'w') as f:
            f.write(str(best_threshold))
        
        print(f"\n💾 Modèle sauvegardé : {LSTM_MODEL_FILE}")
        print(f"💾 Seuil sauvegardé : {threshold_file}")
        
        return self.lstm_model, history
    
    def _find_best_threshold(self, y_true, y_pred_proba):
        """
        Trouve le meilleur seuil pour maximiser F1-Score.
        
        On teste tous les seuils de 0.05 à 0.95 par pas de 0.01
        et on garde celui qui donne le meilleur F1-Score.
        
        Résultat typique : ~0.21 pour LightGBM, ~0.50 pour LSTM
        """
        thresholds = np.arange(0.05, 0.95, 0.01)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def _print_metrics(self, y_true, y_pred, y_pred_proba, model_name):
        """
        Affiche toutes les métriques de performance.
        
        Métriques calculées :
        - Accuracy : % de prédictions correctes
        - Precision : % de vraies positives parmi les prédictions positives
        - Recall : % de vraies positives détectées
        - F1-Score : Moyenne harmonique de Precision et Recall
        - ROC-AUC : Capacité à discriminer les classes
        - Matrice de confusion : TN, FP, FN, TP
        """
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        print(f"\n📈 Métriques - {model_name}")
        print("─" * 40)
        print(f"   Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
        print(f"   Precision : {prec:.4f}")
        print(f"   Recall    : {rec:.4f}")
        print(f"   F1-Score  : {f1:.4f}")
        print(f"   ROC-AUC   : {auc:.4f}")
        
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n📊 Matrice de confusion :")
        print(f"   TN: {cm[0,0]:6d}  |  FP: {cm[0,1]:6d}")
        print(f"   FN: {cm[1,0]:6d}  |  TP: {cm[1,1]:6d}")
        
        print(f"\n📋 Rapport détaillé :")
        print(classification_report(y_true, y_pred,
                                    target_names=['Pas de coupure', 'Coupure'],
                                    zero_division=0))
    
    def train_all(self):
        """
        Entraîne tous les modèles séquentiellement.
        
        Pipeline complet :
        1. Préparation des données (DataPipeline)
        2. Entraînement LightGBM
        3. Entraînement LSTM
        """
        print("\n" + "="*60)
        print("🚀 ENTRAÎNEMENT V6 - SANS SMOTE (Correction Finale)")
        print("="*60)
        
        # Préparer les données
        print("\n1️⃣ Préparation des données...")
        data = self.pipeline.process_for_training(save_processed=True)
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        # Entraîner LightGBM
        print("\n2️⃣ Entraînement LightGBM (sans SMOTE)...")
        self.train_lgbm_improved(X_train, y_train, X_test, y_test)
        
        # Entraîner LSTM
        print("\n3️⃣ Entraînement LSTM...")
        self.train_lstm_improved(X_train, y_train, X_test, y_test)
        
        print("\n" + "="*60)
        print("✅ TOUS LES MODÈLES ENTRAÎNÉS !")
        print("="*60)


def main():
    """
    Fonction principale du script.
    
    Exécutée quand on lance : python scripts/2_train_models.py
    
    Cette fonction crée l'entraîneur et lance l'entraînement complet.
    """
    try:
        trainer = ImprovedModelTrainer()
        trainer.train_all()
        
        print("\n🎉 Entraînement terminé !")
        print(f"📁 Modèles sauvegardés dans : {LGBM_MODEL_FILE.parent}")
    except RuntimeError as e:
        print(f"\n❌ ERREUR FATALE: {e}")


if __name__ == "__main__":
    main()