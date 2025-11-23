"""
Model Training Module with Fixed LightGBM Parameters
=====================================================

OBJECTIF PRINCIPAL :
Ce module entraîne un modèle LightGBM pour prédire les coupures d'électricité.
Il gère le pipeline complet : split des données, SMOTE, entraînement, évaluation, sauvegarde.

CORRECTIONS V2 (PAR RAPPORT À V1) :
- ✅ Paramètres LightGBM ajustés pour éviter l'overfitting sur "quartier_encoded"
- ✅ SMOTE "léger" (20% max au lieu de 50/50) pour préserver les corrélations réelles
- ✅ Régularisation L1/L2 pour pénaliser les splits faciles
- ✅ Arbres moins profonds (max_depth=6 au lieu de 10)

ARCHITECTURE :
Ce fichier est une VERSION ALTERNATIVE du script 2_train_models.py
Il utilise une approche orientée objet (classe ModelTrainer) au lieu de fonctions.

AVANTAGES DE CETTE APPROCHE :
- Encapsulation : Tout est dans la classe ModelTrainer
- État persistant : metrics, model, feature_names stockés
- Réutilisable : Facile de créer plusieurs instances
- Testable : Chaque méthode testable indépendamment

DURÉE : ~2 minutes (SMOTE + entraînement LightGBM)
"""

import os
import json
import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# === CONFIGURATION DU LOGGING ===
# Logging permet de tracer l'exécution sans polluer stdout avec des print()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CLASSE PRINCIPALE : ModelTrainer
# ============================================================================

class ModelTrainer:
    """
    Entraîneur de modèle LightGBM avec importance des features équilibrée.
    
    PHILOSOPHIE DE CONCEPTION :
    Cette classe suit le principe "Single Responsibility" :
    - Elle gère UNIQUEMENT l'entraînement du modèle LightGBM
    - Pas de préprocessing (fait par un autre module)
    - Pas de déploiement (fait par un autre module)
    
    PATTERN UTILISÉ : Template Method
    Le workflow d'entraînement est fixe :
    1. prepare_stratified_split() → Découpe train/test
    2. apply_light_smote() → Équilibre les classes
    3. train() → Entraîne le modèle
    4. evaluate() → Évalue les performances
    5. save() → Sauvegarde modèle + métriques
    
    ÉTAT INTERNE (ATTRIBUTS) :
    - model_dir : Dossier de sauvegarde
    - model : Modèle LightGBM entraîné
    - feature_names : Liste des noms de features
    - metrics : Dictionnaire des métriques (accuracy, f1, etc.)
    
    EXEMPLE D'UTILISATION :
    ```python
    trainer = ModelTrainer(model_dir='models')
    train_df, test_df = trainer.prepare_stratified_split(df)
    trainer.train(train_df)
    trainer.evaluate(test_df)
    trainer.save()
    ```
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialise l'entraîneur.
        
        Args:
            model_dir : Dossier où sauvegarder le modèle (défaut: 'models/')
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)  # Créer le dossier si inexistant
        
        # État initial
        self.model = None           # Sera rempli après train()
        self.feature_names = None   # Sera rempli après train()
        self.metrics = {}           # Sera rempli après evaluate()
    
    def prepare_stratified_split(
        self, 
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Découpe les données en train/test de manière CHRONOLOGIQUE.
        
        POURQUOI CHRONOLOGIQUE (ET PAS ALÉATOIRE) ?
        En séries temporelles, on DOIT respecter l'ordre temporel :
        - Entraînement sur le passé (80% des données)
        - Test sur le futur (20% des données)
        
        Si on fait un split aléatoire, on "triche" :
        - Le modèle voit des données du futur pendant l'entraînement
        - Les performances sont artificiellement gonflées
        - En production, le modèle sera moins bon
        
        EXEMPLE :
        Données : Jan → Déc 2023 (12 mois)
        Split 80/20 :
        - Train : Jan → Oct (10 mois)
        - Test : Nov → Déc (2 mois)
        
        STRATIFICATION :
        Même si le split est chronologique, on vérifie que le taux de coupures
        est similaire dans train et test (affichage informatif).
        
        Args:
            df : DataFrame avec colonnes 'date' et 'coupure'
            test_size : Proportion du test set (défaut: 0.2 = 20%)
        
        Returns:
            Tuple[train_df, test_df] : Données d'entraînement et de test
        """
        # Trier par date (garantir l'ordre chronologique)
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculer l'index de découpe (80% des données)
        split_idx = int(len(df) * (1 - test_size))
        
        # Split chronologique
        train_df = df.iloc[:split_idx].copy()   # 0 → split_idx
        test_df = df.iloc[split_idx:].copy()    # split_idx → fin
        
        # Afficher les statistiques du split
        logger.info(f"Train size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"Test size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        logger.info(f"Train positive rate: {train_df['coupure'].mean()*100:.2f}%")
        logger.info(f"Test positive rate: {test_df['coupure'].mean()*100:.2f}%")
        
        return train_df, test_df
    
    def apply_light_smote(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applique SMOTE "léger" pour réduire le déséquilibre (SANS équilibrer complètement).
        
        SMOTE (Synthetic Minority Over-sampling Technique) :
        Algorithme qui crée des données synthétiques de la classe minoritaire.
        
        COMMENT ÇA MARCHE ?
        1. Pour chaque échantillon minoritaire (coupure=1)
        2. Trouver ses k voisins les plus proches (k=5)
        3. Créer un nouvel échantillon entre l'original et un voisin aléatoire
        4. Répéter jusqu'à atteindre le ratio cible
        
        SMOTE "LÉGER" VS "COMPLET" :
        ❌ SMOTE complet : 50/50 (autant de 0 que de 1)
        ✅ SMOTE léger : 20% max de 1 (ou 2× le ratio original)
        
        POURQUOI "LÉGER" ?
        - Préserve mieux les corrélations réelles des données
        - Évite l'overfitting sur les données synthétiques
        - Plus proche de la distribution réelle en production
        
        EXEMPLE :
        Avant SMOTE : 93% classe 0, 7% classe 1 (déséquilibré)
        Après SMOTE léger : 85% classe 0, 15% classe 1 (moins déséquilibré)
        
        QUAND SMOTE N'EST PAS APPLIQUÉ :
        Si le ratio est déjà >= 20%, on ne fait rien (logger.info "No SMOTE needed")
        
        Args:
            X : Features (DataFrame)
            y : Target (Series avec 0 et 1)
        
        Returns:
            Tuple[X_resampled, y_resampled] : Données rééchantillonnées
        """
        # Calculer le ratio original
        original_ratio = y.sum() / len(y)  # Proportion de 1
        
        # Calculer le ratio cible (20% max, ou 2× l'original)
        target_ratio = min(0.2, original_ratio * 2)
        
        # Calculer le nombre d'échantillons nécessaires
        n_minority = y.sum()                    # Nombre de 1
        n_majority = len(y) - n_minority        # Nombre de 0
        target_minority = int(n_majority * target_ratio / (1 - target_ratio))
        
        # Vérifier si SMOTE est nécessaire
        if target_minority <= n_minority:
            logger.info("No SMOTE needed (ratio already good)")
            return X, y
        
        # Initialiser SMOTE
        smote = SMOTE(
            sampling_strategy=target_minority / n_majority,  # Ratio cible
            random_state=42,     # Reproductibilité
            k_neighbors=5        # Nombre de voisins (standard)
        )
        
        # Appliquer SMOTE
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Afficher les statistiques
        logger.info(f"Before SMOTE: {y.sum()}/{len(y)} = {original_ratio*100:.2f}%")
        logger.info(f"After SMOTE: {y_resampled.sum()}/{len(y_resampled)} = {y_resampled.mean()*100:.2f}%")
        
        return X_resampled, y_resampled
    
    def train(self, train_df: pd.DataFrame) -> Dict:
        """
        Entraîne le modèle LightGBM avec paramètres corrigés (V2).
        
        CORRECTIONS V2 PAR RAPPORT À V1 :
        Le problème V1 : Le modèle se focalisait trop sur "quartier_encoded"
        → Dakar-Plateau prédit comme plus risqué que Guediawaye (INVERSÉ)
        
        Causes identifiées :
        1. SMOTE trop agressif (50/50) → mélangeait les quartiers
        2. Arbres trop profonds (max_depth=10) → overfitting facile
        3. Pas de régularisation → pas de pénalité pour splits simples
        
        Solutions V2 :
        ✅ SMOTE léger (20% max) → préserve les corrélations
        ✅ max_depth=6 (au lieu de 10) → arbres moins profonds
        ✅ min_child_samples=50 (au lieu de 20) → splits plus robustes
        ✅ reg_alpha=0.1, reg_lambda=0.1 → régularisation L1/L2
        ✅ min_split_gain=0.01 → évite les splits trop faciles
        
        PARAMÈTRES LIGHTGBM EXPLIQUÉS :
        
        - objective='binary' : Classification binaire (0 ou 1)
        - metric='auc' : Optimiser l'aire sous la courbe ROC
        - boosting_type='gbdt' : Gradient Boosting (standard)
        
        - num_leaves=31 : Nombre de feuilles par arbre
          (réduit de 60 → 31 pour moins d'overfitting)
        
        - max_depth=6 : Profondeur maximale des arbres
          (réduit de 10 → 6 pour éviter mémorisation de "quartier")
        
        - learning_rate=0.05 : Taux d'apprentissage
          (augmenté de 0.03 → 0.05 pour convergence plus rapide)
        
        - feature_fraction=0.9 : Proportion de features par arbre (90%)
          (augmenté de 0.8 → 0.9 pour utiliser plus d'info)
        
        - bagging_fraction=0.9 : Proportion de données par arbre (90%)
          (augmenté pour plus de stabilité)
        
        - min_child_samples=50 : Min échantillons par feuille
          (augmenté de 20 → 50 pour éviter feuilles trop spécifiques)
        
        - min_split_gain=0.01 : Gain minimum pour créer un split
          (nouveau en V2, empêche les splits triviaux)
        
        - reg_alpha=0.1 : Régularisation L1 (Lasso)
          (nouveau en V2, pénalise les poids élevés)
        
        - reg_lambda=0.1 : Régularisation L2 (Ridge)
          (nouveau en V2, lisse les poids)
        
        - scale_pos_weight=2.0 : Poids de la classe positive
          (réduit de 10.0 → 2.0 car SMOTE a déjà équilibré)
        
        - n_estimators=500 : Nombre d'arbres
          (réduit de 1000 → 500, early stopping prendra le relais)
        
        Args:
            train_df : DataFrame d'entraînement avec colonnes features + 'coupure' + 'date'
        
        Returns:
            Dict : Métriques d'entraînement (feature_importance)
        """
        logger.info("Starting model training...")
        
        # === ÉTAPE 1 : SÉPARER FEATURES ET TARGET ===
        # Exclure 'coupure' (target) et 'date' (pas une feature)
        feature_cols = [col for col in train_df.columns 
                       if col not in ['coupure', 'date']]
        
        X_train = train_df[feature_cols]
        y_train = train_df['coupure']
        
        # Sauvegarder les noms de features (pour prédictions futures)
        self.feature_names = feature_cols
        
        # === ÉTAPE 2 : APPLIQUER SMOTE LÉGER ===
        X_train_balanced, y_train_balanced = self.apply_light_smote(X_train, y_train)
        
        # === ÉTAPE 3 : DÉFINIR LES PARAMÈTRES LIGHTGBM (V2 CORRIGÉS) ===
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,          # ✅ Réduit pour éviter overfitting
            'max_depth': 6,            # ✅ Arbres moins profonds
            'learning_rate': 0.05,     # ✅ Convergence plus rapide
            'feature_fraction': 0.9,   # ✅ Utilise plus de features
            'bagging_fraction': 0.9,   # ✅ Plus de données par arbre
            'bagging_freq': 5,
            'min_child_samples': 50,   # ✅ Feuilles plus robustes
            'min_split_gain': 0.01,    # ✅ NOUVEAU : Évite splits triviaux
            'reg_alpha': 0.1,          # ✅ NOUVEAU : Régularisation L1
            'reg_lambda': 0.1,         # ✅ NOUVEAU : Régularisation L2
            'scale_pos_weight': 2.0,   # Compense le déséquilibre restant
            'verbose': -1,             # Pas de logs verbeux
            'n_estimators': 500,       # ✅ Moins d'arbres (early stopping)
            'random_state': 42         # Reproductibilité
        }
        
        # === ÉTAPE 4 : CRÉER LE DATASET LIGHTGBM ===
        train_data = lgb.Dataset(
            X_train_balanced,
            label=y_train_balanced,
            feature_name=feature_cols  # Noms des colonnes (pour importance)
        )
        
        # === ÉTAPE 5 : ENTRAÎNER LE MODÈLE ===
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data],  # Validation sur train (pour early stopping)
            callbacks=[lgb.early_stopping(stopping_rounds=50)]  # Stop si pas d'amélioration
        )
        
        logger.info("Training completed")
        
        # === ÉTAPE 6 : CALCULER L'IMPORTANCE DES FEATURES ===
        # importance_type='gain' : Combien chaque feature améliore le modèle
        importance = self.model.feature_importance(importance_type='gain')
        self.metrics['feature_importance'] = dict(zip(feature_cols, importance.tolist()))
        
        # Afficher le top 5
        logger.info("\n=== TOP 5 FEATURES BY IMPORTANCE ===")
        for feat, imp in sorted(
            self.metrics['feature_importance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]:
            logger.info(f"{feat}: {imp:.0f}")
        
        return self.metrics
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """
        Évalue le modèle sur le test set.
        
        MÉTRIQUES CALCULÉES :
        1. Accuracy : % de prédictions correctes (attention aux données déséquilibrées)
        2. Precision : % de vrais positifs parmi les prédictions positives
        3. Recall : % de vrais positifs détectés
        4. F1-Score : Moyenne harmonique de Precision et Recall
        5. ROC-AUC : Capacité à discriminer les classes
        6. Confusion Matrix : TN, FP, FN, TP
        
        SEUIL DE DÉCISION :
        On utilise 0.5 par défaut (proba >= 0.5 → prédiction=1)
        En production, on pourrait optimiser ce seuil selon F1-Score.
        
        Args:
            test_df : DataFrame de test avec colonnes features + 'coupure' + 'date'
        
        Returns:
            Dict : Toutes les métriques d'évaluation
        """
        logger.info("Evaluating model on test set...")
        
        # === ÉTAPE 1 : PRÉPARER LES DONNÉES ===
        feature_cols = [col for col in test_df.columns 
                       if col not in ['coupure', 'date']]
        X_test = test_df[feature_cols]
        y_test = test_df['coupure']
        
        # === ÉTAPE 2 : PRÉDICTIONS ===
        # predict() retourne des probabilités (0.0 à 1.0)
        y_pred_proba = self.model.predict(X_test)
        
        # Convertir en prédictions binaires (0 ou 1) avec seuil 0.5
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # === ÉTAPE 3 : CALCULER LES MÉTRIQUES ===
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Sauvegarder dans l'état de la classe
        self.metrics.update(metrics)
        
        # === ÉTAPE 4 : AFFICHER LES RÉSULTATS ===
        logger.info(f"\n=== TEST SET METRICS ===")
        logger.info(f"Accuracy  : {metrics['accuracy']:.4f}")
        logger.info(f"Precision : {metrics['precision']:.4f}")
        logger.info(f"Recall    : {metrics['recall']:.4f}")
        logger.info(f"F1-Score  : {metrics['f1']:.4f}")
        logger.info(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def save(self):
        """
        Sauvegarde le modèle et les métriques.
        
        FICHIERS GÉNÉRÉS :
        1. lightgbm_model.txt : Modèle LightGBM (format texte)
        2. metrics.json : Toutes les métriques (format JSON)
        
        FORMAT LIGHTGBM :
        LightGBM sauvegarde les modèles en .txt (pas pickle).
        Avantages :
        - Lisible par l'humain (on peut voir les arbres)
        - Compatible entre versions de LightGBM
        - Taille réduite
        
        FORMAT MÉTRIQUES :
        JSON pour facilité de lecture et interopérabilité.
        Peut être lu par n'importe quel langage.
        """
        # Sauvegarder le modèle
        model_path = os.path.join(self.model_dir, 'lightgbm_model.txt')
        self.model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Sauvegarder les métriques
        metrics_path = os.path.join(self.model_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Pipeline principal d'entraînement.
    
    WORKFLOW COMPLET :
    1. Charger les données prétraitées (engineered_features.csv)
    2. Initialiser le ModelTrainer
    3. Split train/test chronologique
    4. Entraîner le modèle (avec SMOTE léger)
    5. Évaluer sur le test set
    6. Sauvegarder modèle + métriques
    
    FICHIER D'ENTRÉE :
    data/processed/engineered_features.csv
    (Généré par un script de feature engineering)
    
    FICHIERS DE SORTIE :
    - models/lightgbm_model.txt
    - models/metrics.json
    
    DURÉE TYPIQUE : ~2 minutes
    """
    logger.info("=== TRAINING PIPELINE START ===")
    
    # === ÉTAPE 1 : CHARGER LES DONNÉES ===
    df = pd.read_csv('data/processed/engineered_features.csv', parse_dates=['date'])
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # === ÉTAPE 2 : INITIALISER L'ENTRAÎNEUR ===
    trainer = ModelTrainer()
    
    # === ÉTAPE 3 : SPLIT TRAIN/TEST ===
    train_df, test_df = trainer.prepare_stratified_split(df)
    
    # === ÉTAPE 4 : ENTRAÎNER ===
    trainer.train(train_df)
    
    # === ÉTAPE 5 : ÉVALUER ===
    trainer.evaluate(test_df)
    
    # === ÉTAPE 6 : SAUVEGARDER ===
    trainer.save()
    
    logger.info("=== TRAINING PIPELINE COMPLETE ===")


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == '__main__':
    """
    Point d'entrée quand on exécute : python model_trainer.py
    
    PRÉREQUIS :
    - Fichier data/processed/engineered_features.csv existant
    - Bibliothèques installées (pip install -r requirements.txt)
    
    UTILISATION :
    python model_trainer.py
    
    SORTIE ATTENDUE :
    - Logs d'entraînement dans la console
    - Modèle sauvegardé dans models/lightgbm_model.txt
    - Métriques sauvegardées dans models/metrics.json
    """
    main()


# ============================================================================
# NOTES PÉDAGOGIQUES POUR DATA SCIENTIST JUNIOR
# ============================================================================

"""
📚 CONCEPTS CLÉS À RETENIR :

1. PROGRAMMATION ORIENTÉE OBJET (POO)
   -----------------------------------
   Ce module utilise une classe ModelTrainer au lieu de fonctions.
   
   Avantages :
   - Encapsulation : État (model, metrics) + comportement (train, evaluate)
   - Réutilisabilité : Facile de créer plusieurs instances
   - Organisation : Code plus structuré
   - Testabilité : Chaque méthode testable indépendamment
   
   Comparaison :
   ❌ Fonctions : train(), evaluate(), save() → État global (pas propre)
   ✅ Classe : trainer.train(), trainer.evaluate(), trainer.save() → État encapsulé

2. SPLIT CHRONOLOGIQUE (VS ALÉATOIRE)
   -----------------------------------
   En séries temporelles, TOUJOURS splitter chronologiquement.
   
   Pourquoi ?
   - On prédit le FUTUR à partir du PASSÉ
   - Split aléatoire = triche (voir des données futures)
   - Performances artificiellement gonflées
   
   Règle d'or :
   Train = Passé (80% des données)
   Test = Futur (20% des données)

3. SMOTE - POURQUOI "LÉGER" ?
   ---------------------------
   SMOTE crée des données synthétiques pour équilibrer les classes.
   
   Problème : SMOTE trop agressif (50/50) :
   - Crée trop de données artificielles
   - Dilue les patterns réels
   - Overfitting sur les données synthétiques
   - Inverse les corrélations (Guediawaye < Dakar-Plateau)
   
   Solution : SMOTE léger (20% max) :
   - Réduit le déséquilibre SANS tout inverser
   - Préserve les corrélations réelles
   - Meilleure généralisation
   
   Analogie :
   SMOTE complet = Augmenter le volume à fond (saturation)
   SMOTE léger = Augmenter juste ce qu'il faut (équilibre)

4. PARAMÈTRES LIGHTGBM - GUIDE COMPLET
   ------------------------------------
   LightGBM a 100+ paramètres. Voici les plus importants :
   
   CONTRÔLE DE LA COMPLEXITÉ (éviter overfitting) :
   - num_leaves : Nombre de feuilles par arbre (↓ = moins complexe)
   - max_depth : Profondeur des arbres (↓ = moins complexe)
   - min_child_samples : Min échantillons par feuille (↑ = plus robuste)
   - min_split_gain : Gain min pour un split (↑ = moins de splits)
   
   RÉGULARISATION :
   - reg_alpha : L1 regularization (Lasso, pénalise poids élevés)
   - reg_lambda : L2 regularization (Ridge, lisse les poids)
   
   SAMPLING :
   - feature_fraction : % de features par arbre (↓ = plus diverse)
   - bagging_fraction : % de données par arbre (↓ = plus diverse)
   
   APPRENTISSAGE :
   - learning_rate : Taux d'apprentissage (↓ = plus lent mais précis)
   - n_estimators : Nombre d'arbres (early stopping contrôle)
   
   DÉSÉQUILIBRE :
   - scale_pos_weight : Poids de la classe positive (↑ si très déséquilibré)

5. EARLY STOPPING
   ---------------
   Mécanisme qui arrête l'entraînement automatiquement.
   
   Comment ça marche ?
   - Surveille une métrique sur validation set
   - Si pas d'amélioration pendant N itérations (patience)
   - → Arrête l'entraînement et garde le meilleur modèle
   
   Avantages :
   - Évite l'overfitting (arrête avant que le modèle mémorise)
   - Économise du temps (pas besoin de faire 1000 itérations)
   - Trouve le nombre optimal d'arbres automatiquement
   
   Dans notre code :
   lgb.early_stopping(stopping_rounds=50)
   → Arrête si pas d'amélioration pendant 50 itérations

6. IMPORTANCE DES FEATURES
   ------------------------
   Mesure de l'utilité de chaque feature pour le modèle.
   
   Deux types :
   - 'gain' : Amélioration de la perte apportée par la feature (utilisé ici)
   - 'split' : Nombre de fois que la feature est utilisée
   
   Utilité :
   - Comprendre quelles features sont importantes
   - Feature selection (supprimer les inutiles)
   - Interprétabilité (expliquer les prédictions)
   - Debugging (détecter features trop dominantes)
   
   Exemple d'analyse :
   Si "quartier_encoded" a 80% d'importance :
   → Problème ! Le modèle se base trop sur le quartier
   → Ajuster les paramètres (max_depth, régularisation)

7. MÉTRIQUES D'ÉVALUATION - GUIDE COMPLET
   ---------------------------------------
   Chaque métrique révèle un aspect différent du modèle.
   
   ACCURACY (Exactitude) :
   - Formule : (TP + TN) / Total
   - Signification : % de prédictions correctes
   - ⚠️ PIÈGE : Trompeuse sur données déséquilibrées
   - Exemple : 93% accuracy si on prédit toujours "pas de coupure" (93% de 0)
   
   PRECISION (Précision) :
   - Formule : TP / (TP + FP)
   - Signification : % de vraies coupures parmi les prédictions de coupure
   - Usage : Minimiser les fausses alertes
   - Question : "Parmi toutes mes alertes, combien sont vraies ?"
   
   RECALL (Rappel / Sensibilité) :
   - Formule : TP / (TP + FN)
   - Signification : % de coupures réelles détectées
   - Usage : Ne rater aucune coupure
   - Question : "Parmi toutes les coupures, combien ai-je détectées ?"
   
   F1-SCORE :
   - Formule : 2 × (Precision × Recall) / (Precision + Recall)
   - Signification : Moyenne harmonique de Precision et Recall
   - Usage : Équilibre entre Precision et Recall
   - C'est LA métrique pour données déséquilibrées
   
   ROC-AUC (Area Under ROC Curve) :
   - Valeur : 0.5 (aléatoire) à 1.0 (parfait)
   - Signification : Capacité à discriminer les classes
   - Usage : Comparer différents modèles
   - Indépendant du seuil de décision
   
   CONFUSION MATRIX :
   ┌────────────┬──────────┬──────────┐
   │            │ Prédit 0 │ Prédit 1 │
   ├────────────┼──────────┼──────────┤
   │ Réel 0     │    TN    │    FP    │
   │ Réel 1     │    FN    │    TP    │
   └────────────┴──────────┴──────────┘
   
   - TN (True Negative) : Pas de coupure, correctement prédit ✅
   - FP (False Positive) : Fausse alerte 😐
   - FN (False Negative) : Coupure ratée ❌ (LE PIRE)
   - TP (True Positive) : Coupure détectée ✅✅

8. TRADE-OFF PRECISION VS RECALL
   ------------------------------
   On ne peut pas maximiser les deux en même temps.
   
   Seuil bas (ex: 0.1) :
   - Recall élevé (détecte presque toutes les coupures)
   - Precision faible (beaucoup de fausses alertes)
   
   Seuil élevé (ex: 0.9) :
   - Precision élevée (peu de fausses alertes)
   - Recall faible (rate des coupures)
   
   Seuil optimal (ex: 0.5) :
   - F1-Score maximal (compromis)
   
   Choix selon le métier :
   - Médical (cancer) : Recall élevé (ne rater aucun cas)
   - Spam : Precision élevée (ne pas bloquer vrais emails)
   - Notre cas : F1 équilibré (ni trop d'alertes ni trop de ratés)

9. LOGGING VS PRINT
   -----------------
   Ce module utilise logging au lieu de print().
   
   Avantages du logging :
   - Niveaux de sévérité (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   - Timestamps automatiques
   - Filtrage facile (afficher que les ERROR)
   - Sauvegarde dans fichiers
   - Configuration centralisée
   
   Exemple :
   ```python
   # ❌ MAUVAIS
   print("Training started")
   
   # ✅ BON
   logger.info("Training started")
   ```
   
   Configuration :
   ```python
   logging.basicConfig(
       level=logging.INFO,           # Niveau minimum
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('train.log'),  # Fichier
           logging.StreamHandler()            # Console
       ]
   )
   ```

10. TYPE HINTS (ANNOTATIONS DE TYPES)
    ----------------------------------
    Ce module utilise des annotations de types (Python 3.5+).
    
    Exemple :
    ```python
    def train(self, train_df: pd.DataFrame) -> Dict:
        ...
    ```
    
    Signification :
    - train_df: pd.DataFrame → Paramètre doit être un DataFrame
    - -> Dict → Fonction retourne un dictionnaire
    
    Avantages :
    - Documentation automatique (on voit les types attendus)
    - Détection d'erreurs (IDE signale les types incorrects)
    - Meilleure autocomplétion
    - Code plus lisible
    
    Types courants :
    - int, float, str, bool → Types simples
    - List[int] → Liste d'entiers
    - Dict[str, float] → Dict avec clés str, valeurs float
    - Tuple[int, int] → Tuple de 2 entiers
    - Optional[str] → Peut être str ou None

11. PATTERN TEMPLATE METHOD
    ------------------------
    La classe ModelTrainer suit ce pattern de conception.
    
    Principe :
    - Définir le squelette d'un algorithme dans une méthode
    - Déléguer certaines étapes à des sous-méthodes
    - L'ordre est fixe, les détails flexibles
    
    Dans notre cas :
    main() définit le workflow :
    1. Load data
    2. Split train/test (prepare_stratified_split)
    3. Train (train)
    4. Evaluate (evaluate)
    5. Save (save)
    
    Avantages :
    - Structure claire et prévisible
    - Facile de modifier une étape sans casser le reste
    - Testable étape par étape
    
    Variante possible :
    On pourrait créer une classe abstraite avec ces méthodes,
    et des classes enfants (RandomForestTrainer, XGBoostTrainer, etc.)

12. GESTION DES ERREURS (ROBUSTESSE)
    ---------------------------------
    Ce module pourrait être amélioré avec try/except.
    
    Points à protéger :
    - Chargement du CSV (fichier manquant, corrompu)
    - Entraînement (OOM, interruption)
    - Sauvegarde (disque plein, permissions)
    
    Exemple de version robuste :
    ```python
    def save(self):
        try:
            model_path = os.path.join(self.model_dir, 'lightgbm_model.txt')
            self.model.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
        except OSError as e:
            logger.error(f"Failed to save model: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    ```

13. VERSIONING DES MODÈLES
    -----------------------
    En production, il faut versionner les modèles.
    
    Stratégies :
    
    Approche 1 : Timestamp dans le nom
    ```python
    model_path = f'models/lgbm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    ```
    
    Approche 2 : Git pour les modèles
    - Git LFS (Large File Storage)
    - DVC (Data Version Control)
    
    Approche 3 : MLflow
    - Tracking des expériences
    - Registry de modèles
    - Comparaison automatique
    
    Approche 4 : Dossiers numérotés
    ```
    models/
    ├── v1/
    │   ├── model.txt
    │   └── metrics.json
    ├── v2/
    │   ├── model.txt
    │   └── metrics.json
    └── v3/ (current)
    ```

14. REPRODUCTIBILITÉ
    -----------------
    Ce module est reproductible grâce à :
    
    1. random_state=42 partout
    2. Sort chronologique (pas d'aléatoire)
    3. SMOTE avec random_state
    4. LightGBM avec random_state
    
    Pourquoi c'est important ?
    - Debugging : Même résultats = même bug
    - Collaboration : Équipe voit les mêmes résultats
    - Validation : Prouver que les résultats sont solides
    - Recherche : Papiers reproductibles
    
    Checklist reproductibilité :
    ✅ random_state fixé
    ✅ Pas de shuffle aléatoire
    ✅ Versions des librairies fixées (requirements.txt)
    ✅ Code versionné (Git)
    ✅ Documentation complète

15. OPTIMISATION FUTURE
    --------------------
    Améliorations possibles de ce module :
    
    A. Hyperparameter Tuning
       - Optuna, GridSearchCV, RandomSearchCV
       - Trouver les meilleurs paramètres automatiquement
    
    B. Cross-Validation
       - K-fold validation (k=5)
       - Time series split (respecting chronology)
       - Plus robuste que single split
    
    C. Feature Engineering automatique
       - Featuretools, tsfresh
       - Polynomiale features
       - Interactions automatiques
    
    D. Ensemble Methods
       - Stacking (LightGBM + XGBoost + RF)
       - Voting classifier
       - Améliore performances de 1-3%
    
    E. Calibration des probabilités
       - Platt scaling, Isotonic regression
       - Probabilités plus fiables
    
    F. Monitoring en production
       - Data drift detection
       - Model drift detection
       - Alertes automatiques
    
    G. Explainability
       - SHAP values
       - LIME
       - Feature importance locale

16. COMMANDES UTILES
    -----------------
    # Entraîner le modèle
    python model_trainer.py
    
    # Voir les logs en temps réel
    tail -f train.log
    
    # Vérifier le modèle sauvegardé
    ls -lh models/
    
    # Visualiser les métriques
    cat models/metrics.json | python -m json.tool
    
    # Charger le modèle en Python
    import lightgbm as lgb
    model = lgb.Booster(model_file='models/lightgbm_model.txt')
    
    # Comparer plusieurs versions
    diff models/v1/metrics.json models/v2/metrics.json

17. ERREURS COURANTES ET SOLUTIONS
    --------------------------------
    ❌ "FileNotFoundError: engineered_features.csv"
    ✅ Lancer le script de feature engineering d'abord
    
    ❌ "MemoryError during SMOTE"
    ✅ Réduire target_ratio (ex: 0.15 au lieu de 0.2)
    
    ❌ "ValueError: Found array with 0 sample(s)"
    ✅ Vérifier que le CSV n'est pas vide
    
    ❌ "LightGBM: min_data_in_leaf must be at least 1"
    ✅ Réduire min_child_samples si dataset très petit
    
    ❌ Overfitting (train F1 >> test F1)
    ✅ Augmenter régularisation (reg_alpha, reg_lambda)
    ✅ Réduire max_depth
    ✅ Augmenter min_child_samples
    
    ❌ Underfitting (train F1 et test F1 bas)
    ✅ Augmenter n_estimators
    ✅ Réduire learning_rate
    ✅ Augmenter max_depth (mais attention overfitting)
    
    ❌ "quartier" domine feature importance
    ✅ Appliquer les corrections V2 (ce module les a déjà)

18. CHECKLIST AVANT DÉPLOIEMENT
    ----------------------------
    Avant de mettre ce modèle en production :
    
    ✅ F1-Score > 0.60 (seuil minimum)
    ✅ Pas d'overfitting (train F1 ≈ test F1 ± 5%)
    ✅ Feature importance équilibrée (pas de dominance)
    ✅ Confusion matrix acceptable (FN < 30%)
    ✅ Testé sur données out-of-sample
    ✅ Temps d'inférence < 100ms
    ✅ Modèle versionné et tracké
    ✅ Documentation complète
    ✅ Monitoring en place
    ✅ Rollback strategy définie
"""