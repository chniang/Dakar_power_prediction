# Fichier : scripts/2_train_models.py
# Script d'entraînement V6.1 SANS SMOTE (Correction Finale + Fix Keras + Extensions .pkl)
# ========================================================

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

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
    # ✅ CORRECTION : Extensions cohérentes (.pkl et .h5)
    LGBM_MODEL_FILE = Path("models/lgbm_model.pkl")  # ✅ .pkl
    LSTM_MODEL_FILE = Path("models/lstm_model.h5")   # ✅ .h5
    SEQUENCE_LENGTH = 12
    LSTM_EPOCHS = 50

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
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class ImprovedModelTrainer:
    """Entraîneur V6.1 - SANS SMOTE + Fix Keras + Extensions cohérentes"""
    
    def __init__(self):
        if 'DataPipeline' in globals():
            self.pipeline = DataPipeline()
        else:
            raise RuntimeError("La classe DataPipeline n'a pas pu être chargée.")
        
        self.lgbm_model = None
        self.lstm_model = None
    
    def train_lgbm_improved(self, X_train, y_train, X_test, y_test):
        """Entraîne LightGBM SANS SMOTE (correction V6)"""
        print("\n" + "="*60)
        print("🌳 ENTRAÎNEMENT LIGHTGBM V6 (SANS SMOTE)")
        print("="*60)
        
        print(f"\n⚖️ Distribution originale (AUCUN rééchantillonnage) :")
        print(f"   Classe 0 (pas de coupure) : {(y_train == 0).sum():,}")
        print(f"   Classe 1 (coupure)        : {(y_train == 1).sum():,}")
        print(f"   Ratio coupures            : {y_train.mean()*100:.2f}%")
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'min_child_samples': 50,
            'min_split_gain': 0.01,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'scale_pos_weight': 10.0,
            'verbose': -1,
            'n_estimators': 500,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
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
        
        print("\n📊 Importance des features (top 5) :")
        feature_importance = self.lgbm_model.feature_importance(importance_type='gain')
        feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        for idx, row in importance_df.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.0f}")
        
        print("\n📊 Recherche du seuil optimal...")
        y_pred_proba = self.lgbm_model.predict(X_test)
        
        best_threshold = self._find_best_threshold(y_test, y_pred_proba)
        print(f"   🎯 Seuil optimal trouvé : {best_threshold:.3f}")
        
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        self._print_metrics(y_test, y_pred, y_pred_proba, "LightGBM V6 (Sans SMOTE)")
        
        # ✅ CORRECTION : Sauvegarder avec .pkl (joblib)
        LGBM_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.lgbm_model,
            'threshold': best_threshold
        }, LGBM_MODEL_FILE)
        print(f"\n💾 Modèle sauvegardé : {LGBM_MODEL_FILE} (.pkl)")
        
        return self.lgbm_model
    
    def build_improved_lstm(self, input_shape):
        """Construit l'architecture LSTM optimisée - VERSION CORRIGÉE V6.1"""
        model = Sequential([
            Input(shape=input_shape),
            
            LSTM(100, return_sequences=True),
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
        """Entraîne le modèle LSTM avec architecture corrigée"""
        print("\n" + "="*60)
        print("🧠 ENTRAÎNEMENT LSTM V6.1 (Fix Keras)")
        print("="*60)
        
        print(f"\n🔄 Création des séquences (longueur={SEQUENCE_LENGTH})...")
        X_train_seq, y_train_seq = self.pipeline.create_sequences(X_train, y_train, SEQUENCE_LENGTH)
        X_test_seq, y_test_seq = self.pipeline.create_sequences(X_test, y_test, SEQUENCE_LENGTH)
        
        print(f"   Train: {X_train_seq.shape}")
        print(f"   Test:  {X_test_seq.shape}")
        
        neg_count = (y_train_seq == 0).sum()
        pos_count = (y_train_seq == 1).sum()
        class_weight = {0: 1.0, 1: neg_count / pos_count}
        print(f"\n⚖️ Poids des classes : {class_weight}")
        
        print("\n🏗️ Construction du modèle LSTM (avec Input() explicite)...")
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.lstm_model = self.build_improved_lstm(input_shape)
        
        print("\n📐 Architecture du modèle :")
        self.lstm_model.summary()
        
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
        
        print("\n📊 Évaluation sur le test set :")
        y_pred_proba = self.lstm_model.predict(X_test_seq, verbose=0).flatten()
        
        best_threshold = self._find_best_threshold(y_test_seq, y_pred_proba)
        print(f"   🎯 Seuil optimal trouvé : {best_threshold:.3f}")
        
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        self._print_metrics(y_test_seq, y_pred, y_pred_proba, "LSTM V6.1")
        
        threshold_file = LSTM_MODEL_FILE.parent / "lstm_threshold.txt"
        with open(threshold_file, 'w') as f:
            f.write(str(best_threshold))
        
        print(f"\n💾 Modèle sauvegardé : {LSTM_MODEL_FILE} (.h5)")
        print(f"💾 Seuil sauvegardé : {threshold_file}")
        
        return self.lstm_model, history
    
    def _find_best_threshold(self, y_true, y_pred_proba):
        """Trouve le meilleur seuil pour maximiser F1-Score"""
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
        """Affiche toutes les métriques de performance"""
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
        """Entraîne tous les modèles séquentiellement"""
        print("\n" + "="*60)
        print("🚀 ENTRAÎNEMENT V6.1 - SANS SMOTE + Fix Keras + Extensions .pkl")
        print("="*60)
        
        print("\n1️⃣ Préparation des données...")
        data = self.pipeline.process_for_training(save_processed=True)
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        print("\n2️⃣ Entraînement LightGBM (sans SMOTE)...")
        self.train_lgbm_improved(X_train, y_train, X_test, y_test)
        
        print("\n3️⃣ Entraînement LSTM (avec Input() corrigé)...")
        self.train_lstm_improved(X_train, y_train, X_test, y_test)
        
        print("\n" + "="*60)
        print("✅ TOUS LES MODÈLES ENTRAÎNÉS !")
        print("="*60)


def main():
    """Fonction principale du script"""
    try:
        trainer = ImprovedModelTrainer()
        trainer.train_all()
        
        print("\n🎉 Entraînement terminé !")
        print(f"📁 Modèles sauvegardés dans : {LGBM_MODEL_FILE.parent}")
    except RuntimeError as e:
        print(f"\n❌ ERREUR FATALE: {e}")
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()