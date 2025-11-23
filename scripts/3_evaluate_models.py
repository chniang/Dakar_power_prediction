# Fichier : scripts/3_evaluate_models.py
# Script d'évaluation et comparaison des modèles
# ===============================================
#
# OBJECTIF PRINCIPAL :
# Ce script compare les performances des 2 modèles ML (LightGBM vs LSTM)
# en calculant leurs métriques et en générant des visualisations.
#
# FONCTIONNALITÉS :
# 1. Charge les modèles entraînés + leurs seuils optimaux
# 2. Fait des prédictions sur le test set
# 3. Aligne les données (important pour LSTM qui perd SEQUENCE_LENGTH échantillons)
# 4. Calcule toutes les métriques (Accuracy, Precision, Recall, F1, ROC-AUC)
# 5. Génère 3 graphiques de comparaison (Confusion Matrix, ROC, Precision-Recall)
# 6. Sauvegarde un rapport texte avec recommandations
#
# DURÉE : ~30 secondes
#
# UTILISATION :
# python scripts/3_evaluate_models.py                  # Affiche les graphiques
# python scripts/3_evaluate_models.py --no-plots       # Sans graphiques
# python scripts/3_evaluate_models.py --save-plots     # Sauvegarde PNG

import sys
from pathlib import Path
import argparse
import warnings
from datetime import datetime

# Suppression des warnings (évite le bruit dans la console)
warnings.filterwarnings('ignore')

# Ajout du répertoire racine au path Python (pour importer src/)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Bibliothèques externes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tabulate import tabulate

# Métriques scikit-learn
from sklearn.metrics import (
    confusion_matrix,           # Matrice TN/FP/FN/TP
    roc_curve, auc,             # Courbe ROC et AUC
    precision_recall_curve,     # Courbe Precision-Recall
    average_precision_score,    # Aire sous PR curve
    accuracy_score,             # % de prédictions correctes
    precision_score,            # % de vrais positifs parmi les prédictions positives
    recall_score,               # % de vrais positifs détectés
    f1_score,                   # Moyenne harmonique Precision/Recall
    roc_auc_score               # Aire sous ROC curve
)

# Modules internes du projet
from src.data_pipeline import DataPipeline
from src.config import LGBM_MODEL_FILE, LSTM_MODEL_FILE, SEQUENCE_LENGTH


# ============================================================================
# SECTION 1 : VISUALISATIONS
# ============================================================================

def plot_confusion_matrices(y_true, y_pred_lgbm, y_pred_lstm, save_path=None):
    """
    Affiche les matrices de confusion des 2 modèles côte à côte.
    
    MATRICE DE CONFUSION :
    Une grille 2×2 qui montre les 4 types de prédictions possibles :
    
    ┌────────────────┬──────────────────┬──────────────────┐
    │                │  Prédit: Pas de  │  Prédit: Coupure │
    │                │     Coupure      │                  │
    ├────────────────┼──────────────────┼──────────────────┤
    │ Réel: Pas de   │   TN (Vrai Nég)  │   FP (Faux Pos)  │
    │    Coupure     │   ✅ Correct     │   ❌ Fausse      │
    │                │                  │      Alerte      │
    ├────────────────┼──────────────────┼──────────────────┤
    │ Réel: Coupure  │   FN (Faux Nég)  │   TP (Vrai Pos)  │
    │                │   ❌ Raté        │   ✅ Correct     │
    ├────────────────┴──────────────────┴──────────────────┤
    
    POURQUOI C'EST IMPORTANT :
    - TN/TP (diagonale) : Bonnes prédictions → on veut maximiser
    - FP : Fausse alerte → Gênant mais pas grave
    - FN : Coupure ratée → TRÈS GRAVE (pas de prévention)
    
    Dans notre cas, FN est le pire car ne pas prévenir d'une coupure
    a plus d'impact que prévoir une coupure qui n'arrive pas.
    
    Args:
        y_true : Vraies étiquettes (0=pas de coupure, 1=coupure)
        y_pred_lgbm : Prédictions LightGBM (0 ou 1)
        y_pred_lstm : Prédictions LSTM (0 ou 1)
        save_path : Chemin pour sauvegarder le graphique (None = affichage)
    """
    
    # Créer une figure avec 2 sous-graphiques (1 ligne, 2 colonnes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- GRAPHIQUE 1 : LightGBM ---
    cm_lgbm = confusion_matrix(y_true, y_pred_lgbm)
    sns.heatmap(
        cm_lgbm,                    # Matrice à afficher
        annot=True,                 # Afficher les nombres dans les cases
        fmt='d',                    # Format entier (pas de décimales)
        cmap='Blues',               # Palette de couleurs bleues
        ax=axes[0],                 # Premier sous-graphique
        cbar=False,                 # Pas de barre de couleur
        linewidths=.5,              # Lignes fines entre les cases
        linecolor='lightgray'
    )
    axes[0].set_title('LightGBM - Matrice de Confusion', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Prédiction')
    axes[0].set_ylabel('Réel')
    axes[0].set_xticklabels(['Pas de coupure', 'Coupure'])
    axes[0].set_yticklabels(['Pas de coupure', 'Coupure'])
    
    # --- GRAPHIQUE 2 : LSTM ---
    cm_lstm = confusion_matrix(y_true, y_pred_lstm)
    sns.heatmap(
        cm_lstm,
        annot=True,
        fmt='d',
        cmap='Oranges',             # Palette orangée pour différencier
        ax=axes[1],                 # Deuxième sous-graphique
        cbar=False,
        linewidths=.5,
        linecolor='lightgray'
    )
    axes[1].set_title('LSTM - Matrice de Confusion', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Prédiction')
    axes[1].set_ylabel('Réel')
    axes[1].set_xticklabels(['Pas de coupure', 'Coupure'])
    axes[1].set_yticklabels(['Pas de coupure', 'Coupure'])
    
    # Titre global
    plt.suptitle('Comparaison des Matrices de Confusion', 
                 fontsize=16, fontweight='heavy', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Sauvegarde ou affichage
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Matrices de confusion sauvegardées : {save_path}")
    
    # Afficher si pas de sauvegarde OU si --save-plots ET on veut afficher
    if not save_path or not argparse.ArgumentParser().parse_args().save_plots:
        plt.show()


def plot_roc_curves(y_true, y_proba_lgbm, y_proba_lstm, save_path=None):
    """
    Affiche les courbes ROC (Receiver Operating Characteristic).
    
    COURBE ROC - QU'EST-CE QUE C'EST ?
    C'est un graphique qui montre le compromis entre :
    - TPR (True Positive Rate) = Recall = Sensibilité
    - FPR (False Positive Rate) = Taux de fausses alertes
    
    COMMENT ÇA MARCHE :
    1. On fait varier le seuil de 0 à 1
    2. Pour chaque seuil, on calcule TPR et FPR
    3. On trace la courbe TPR vs FPR
    
    INTERPRÉTATION :
    - Courbe proche du coin supérieur gauche = BON (TPR élevé, FPR faible)
    - Courbe diagonale = MAUVAIS (modèle aléatoire)
    - AUC (Aire sous la courbe) résume la performance :
      * AUC = 1.0 : Parfait
      * AUC = 0.9 : Excellent
      * AUC = 0.8 : Très bon
      * AUC = 0.7 : Bon
      * AUC = 0.5 : Aléatoire (inutile)
    
    EXEMPLE :
    Si AUC = 0.92, ça veut dire qu'il y a 92% de chance que le modèle
    classe une coupure réelle avec un score plus élevé qu'une non-coupure.
    
    Args:
        y_true : Vraies étiquettes
        y_proba_lgbm : Probabilités prédites par LightGBM (0.0 à 1.0)
        y_proba_lstm : Probabilités prédites par LSTM (0.0 à 1.0)
        save_path : Chemin pour sauvegarder
    """
    
    # Calculer les courbes ROC pour chaque modèle
    fpr_lgbm, tpr_lgbm, _ = roc_curve(y_true, y_proba_lgbm)
    roc_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)
    
    fpr_lstm, tpr_lstm, _ = roc_curve(y_true, y_proba_lstm)
    roc_auc_lstm = auc(fpr_lstm, tpr_lstm)
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    
    # Tracer LightGBM
    plt.plot(fpr_lgbm, tpr_lgbm, 
             color='blue', lw=2, 
             label=f'LightGBM (AUC = {roc_auc_lgbm:.3f})')
    
    # Tracer LSTM
    plt.plot(fpr_lstm, tpr_lstm, 
             color='orange', lw=2, 
             label=f'LSTM (AUC = {roc_auc_lstm:.3f})')
    
    # Ligne de référence (modèle aléatoire)
    plt.plot([0, 1], [0, 1], 
             color='gray', lw=1, linestyle='--', 
             label='Aléatoire (AUC = 0.500)')
    
    # Configuration des axes
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
    plt.title('Courbes ROC - Comparaison des Modèles', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    # Sauvegarde ou affichage
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📈 Courbes ROC sauvegardées : {save_path}")
    
    if not save_path or not argparse.ArgumentParser().parse_args().save_plots:
        plt.show()


def plot_precision_recall_curves(y_true, y_proba_lgbm, y_proba_lstm, save_path=None):
    """
    Affiche les courbes Precision-Recall.
    
    COURBE PRECISION-RECALL - POURQUOI L'UTILISER ?
    Contrairement à la courbe ROC, la courbe PR est plus informative
    pour les datasets DÉSÉQUILIBRÉS (comme le nôtre : 93% classe 0, 7% classe 1).
    
    DIFFÉRENCE AVEC ROC :
    - ROC utilise FPR (faux positifs / tous les négatifs)
    - PR utilise Precision (vrais positifs / tous les prédits positifs)
    
    Avec 93% de classe 0, même beaucoup de FP restent un petit FPR,
    mais affectent énormément la Precision. La courbe PR révèle mieux
    ce problème.
    
    INTERPRÉTATION :
    - Courbe proche du coin supérieur droit = BON
    - AP (Average Precision) résume la performance :
      * AP = 1.0 : Parfait
      * AP > 0.5 : Bon sur données déséquilibrées
      * AP < baseline : Mauvais
    
    BASELINE :
    C'est la ligne de référence qui représente un classifieur aléatoire.
    Baseline = proportion de la classe positive (ici ~7%).
    Un bon modèle doit avoir AP >> baseline.
    
    Args:
        y_true : Vraies étiquettes
        y_proba_lgbm : Probabilités LightGBM
        y_proba_lstm : Probabilités LSTM
        save_path : Chemin pour sauvegarder
    """
    
    # Calculer les courbes Precision-Recall
    precision_lgbm, recall_lgbm, _ = precision_recall_curve(y_true, y_proba_lgbm)
    ap_lgbm = average_precision_score(y_true, y_proba_lgbm)
    
    precision_lstm, recall_lstm, _ = precision_recall_curve(y_true, y_proba_lstm)
    ap_lstm = average_precision_score(y_true, y_proba_lstm)
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    
    # Tracer LightGBM
    plt.plot(recall_lgbm, precision_lgbm, 
             color='blue', lw=2,
             label=f'LightGBM (AP = {ap_lgbm:.3f})')
    
    # Tracer LSTM
    plt.plot(recall_lstm, precision_lstm, 
             color='orange', lw=2,
             label=f'LSTM (AP = {ap_lstm:.3f})')
    
    # Ligne de base (proportion de coupures dans les données)
    baseline = y_true.sum() / len(y_true)
    plt.axhline(y=baseline, 
                color='gray', linestyle='--', lw=1,
                label=f'Baseline (AP = {baseline:.3f})')
    
    # Configuration des axes
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensibilité)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Courbes Precision-Recall - Comparaison des Modèles', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    
    # Sauvegarde ou affichage
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📈 Courbes Precision-Recall sauvegardées : {save_path}")
    
    if not save_path or not argparse.ArgumentParser().parse_args().save_plots:
        plt.show()


# ============================================================================
# SECTION 2 : GÉNÉRATION DE RAPPORTS
# ============================================================================

def generate_comparison_table(metrics_lgbm, metrics_lstm):
    """
    Génère un tableau comparatif des métriques.
    
    STRUCTURE DU TABLEAU :
    ┌───────────┬──────────┬──────────┬────────────┐
    │ Métrique  │ LightGBM │   LSTM   │ Différence │
    ├───────────┼──────────┼──────────┼────────────┤
    │ Accuracy  │  0.9234  │  0.8945  │  +0.0289   │
    │ Precision │  0.8123  │  0.7654  │  +0.0469   │
    │    ...    │   ...    │   ...    │    ...     │
    └───────────┴──────────┴──────────┴────────────┘
    
    COLONNE "DIFFÉRENCE" :
    - Valeur positive → LightGBM meilleur
    - Valeur négative → LSTM meilleur
    - Permet de voir rapidement les écarts
    
    Args:
        metrics_lgbm : Dict des métriques LightGBM
        metrics_lstm : Dict des métriques LSTM
    
    Returns:
        DataFrame avec 3 colonnes : LightGBM, LSTM, Différence
    """
    
    # Créer le DataFrame à partir des dictionnaires
    comparison = pd.DataFrame({
        'LightGBM': pd.Series(metrics_lgbm),
        'LSTM': pd.Series(metrics_lstm)
    })
    
    # Calculer la différence (positif = LightGBM meilleur)
    comparison['Différence'] = comparison['LightGBM'] - comparison['LSTM']
    
    return comparison


def save_evaluation_report(comparison_df, output_dir):
    """
    Sauvegarde un rapport d'évaluation complet en fichier texte.
    
    STRUCTURE DU RAPPORT :
    1. En-tête avec date/heure
    2. Tableau de comparaison des métriques
    3. Analyse détaillée :
       - Meilleur modèle pour chaque métrique
       - Interprétation des résultats
    4. Recommandation finale basée sur F1-Score
    
    POURQUOI F1-SCORE POUR LA RECOMMANDATION ?
    Le F1-Score est la métrique la plus équilibrée pour notre cas :
    - Il combine Precision (éviter fausses alertes) et Recall (détecter coupures)
    - Il pénalise les modèles déséquilibrés (bon en Precision mais mauvais en Recall)
    - C'est le standard pour les problèmes de classification déséquilibrée
    
    Args:
        comparison_df : DataFrame de comparaison (de generate_comparison_table)
        output_dir : Dossier où sauvegarder le rapport
    """
    
    # Créer le nom du fichier avec timestamp
    report_path = output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # === EN-TÊTE ===
        f.write("="*70 + "\n")
        f.write("RAPPORT D'ÉVALUATION ET DE COMPARAISON DES MODÈLES\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        # === TABLEAU DE COMPARAISON ===
        f.write("COMPARAISON DES MÉTRIQUES\n")
        f.write("-" * 70 + "\n")
        # tabulate transforme le DataFrame en tableau ASCII joliment formaté
        f.write(tabulate(comparison_df.round(6), headers='keys', tablefmt='fancy_grid')) 
        f.write("\n\n")
        
        # === ANALYSE DES PERFORMANCES ===
        f.write("ANALYSE DES PERFORMANCES\n")
        f.write("-" * 70 + "\n")
        
        # Déterminer le meilleur modèle pour chaque métrique
        best_accuracy = "LightGBM" if comparison_df.loc['Accuracy', 'LightGBM'] > comparison_df.loc['Accuracy', 'LSTM'] else "LSTM"
        best_precision = "LightGBM" if comparison_df.loc['Precision', 'LightGBM'] > comparison_df.loc['Precision', 'LSTM'] else "LSTM"
        best_recall = "LightGBM" if comparison_df.loc['Recall', 'LightGBM'] > comparison_df.loc['Recall', 'LSTM'] else "LSTM"
        best_f1 = "LightGBM" if comparison_df.loc['F1-Score', 'LightGBM'] > comparison_df.loc['F1-Score', 'LSTM'] else "LSTM"
        best_auc = "LightGBM" if comparison_df.loc['ROC-AUC', 'LightGBM'] > comparison_df.loc['ROC-AUC', 'LSTM'] else "LSTM"
        
        # Afficher le meilleur modèle pour chaque métrique
        f.write(f"• Meilleure Accuracy (Générale)  : {best_accuracy} ({comparison_df.loc['Accuracy', best_accuracy]:.6f})\n")
        f.write(f"• Meilleure Precision (Faux Positifs) : {best_precision} ({comparison_df.loc['Precision', best_precision]:.6f})\n")
        f.write(f"• Meilleur Recall (Vrais Positifs)  : {best_recall} ({comparison_df.loc['Recall', best_recall]:.6f})\n")
        f.write(f"• Meilleur F1-Score (Équilibre)  : {best_f1} ({comparison_df.loc['F1-Score', best_f1]:.6f})\n")
        f.write(f"• Meilleur ROC-AUC (Discrimination)  : {best_auc} ({comparison_df.loc['ROC-AUC', best_auc]:.6f})\n\n")
        
        # === RECOMMANDATION FINALE ===
        f.write("CONCLUSION ET RECOMMANDATION\n")
        f.write("-" * 70 + "\n")
        
        # Recommandation basée sur F1-Score (métrique la plus importante)
        if comparison_df.loc['F1-Score', 'LightGBM'] > comparison_df.loc['F1-Score', 'LSTM']:
            f.write("✅ RECOMMANDATION : Utiliser LightGBM comme modèle principal.\n")
            f.write("   LightGBM offre un meilleur équilibre entre précision et rappel (F1-Score), ")
            f.write("ce qui est critique pour la détection de coupures dans des données déséquilibrées.\n")
            f.write(f"   L'écart de F1-Score est de {comparison_df.loc['F1-Score', 'Différence']:.6f} en faveur de LightGBM.\n")
        elif comparison_df.loc['F1-Score', 'LSTM'] > comparison_df.loc['F1-Score', 'LightGBM']:
            f.write("✅ RECOMMANDATION : Utiliser LSTM comme modèle principal.\n")
            f.write("   LSTM offre de meilleures performances globales (F1-Score), ")
            f.write("montrant sa capacité à capturer des dépendances temporelles pertinentes.\n")
            f.write(f"   L'écart de F1-Score est de {abs(comparison_df.loc['F1-Score', 'Différence']):.6f} en faveur de LSTM.\n")
        else:
             f.write("⚠️ RECOMMANDATION : Les modèles ont des performances F1-Score très similaires. ")
             f.write("Choisir en fonction des contraintes de déploiement (vitesse, simplicité, mémoire).\n")
            
        f.write("\n" + "="*70 + "\n")
    
    print(f"\n📄 Rapport d'évaluation sauvegardé : {report_path}")


# ============================================================================
# SECTION 3 : FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale du script d'évaluation.
    
    WORKFLOW COMPLET :
    1. Parser les arguments de ligne de commande
    2. Charger les données de test (via DataPipeline)
    3. Charger les modèles entraînés + leurs seuils
    4. Préparer les données séquentielles pour LSTM
    5. Faire les prédictions avec les 2 modèles
    6. Aligner les données (problème : LSTM perd SEQUENCE_LENGTH échantillons)
    7. Calculer toutes les métriques sur données alignées
    8. Générer les graphiques de comparaison
    9. Sauvegarder le rapport texte
    
    PROBLÈME D'ALIGNEMENT :
    LightGBM prédit sur tout X_test (N échantillons)
    LSTM prédit sur X_test_seq (N - SEQUENCE_LENGTH échantillons)
    
    Solution : On coupe les SEQUENCE_LENGTH premières lignes de X_test
    pour avoir la même taille pour les 2 modèles.
    """
    
    # === ÉTAPE 1 : PARSER LES ARGUMENTS ===
    parser = argparse.ArgumentParser(
        description="Évalue et compare les modèles LightGBM et LSTM"
    )
    parser.add_argument(
        '--no-plots', 
        action='store_true',
        help="Ne pas afficher les graphiques"
    )
    parser.add_argument(
        '--save-plots', 
        action='store_true',
        help="Sauvegarder les graphiques (PNG) dans evaluation_results/"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("📊 SCRIPT 3 : ÉVALUATION ET COMPARAISON DES MODÈLES")
    print("="*70)
    
    # Créer le dossier de sortie pour les résultats
    output_dir = project_root / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # === ÉTAPE 2 : CHARGER LES DONNÉES ===
        print("\n1️⃣ Chargement des données de test...")
        pipeline = DataPipeline() 
        
        # Charger les données prétraitées + split train/test
        data = pipeline.process_for_training(save_processed=False)
        
        X_test = data['X_test']
        y_test = pd.Series(data['y_test'])  # Convertir en Series pour faciliter l'indexation

        print(f"   ✅ {len(X_test):,} échantillons de test chargés")
        
        # === ÉTAPE 3 : CHARGER LES MODÈLES ===
        print("\n2️⃣ Chargement des modèles...")
        
        # LightGBM (sauvegardé avec Joblib)
        lgbm_data = joblib.load(LGBM_MODEL_FILE)
        lgbm_model = lgbm_data['model']
        lgbm_threshold = lgbm_data['threshold']
        print(f"   ✅ LightGBM chargé (seuil = {lgbm_threshold:.3f})")
        
        # LSTM (sauvegardé avec Keras)
        lstm_model = tf.keras.models.load_model(LSTM_MODEL_FILE)
        lstm_threshold_file = LSTM_MODEL_FILE.parent / "lstm_threshold.txt"
        with open(lstm_threshold_file, 'r') as f:
            lstm_threshold = float(f.read().strip())
        print(f"   ✅ LSTM chargé (seuil = {lstm_threshold:.3f})")
        
        # === ÉTAPE 4 : PRÉPARER LES SÉQUENCES POUR LSTM ===
        print("\n3️⃣ Préparation des données séquentielles pour LSTM...")
        X_test_seq, y_test_seq = pipeline.create_sequences(
            X_test, y_test.values, sequence_length=SEQUENCE_LENGTH
        )
        print(f"   ✅ {len(X_test_seq):,} séquences créées")
        print(f"   ℹ️ Perte de {SEQUENCE_LENGTH} échantillons (historique)")

        # === ÉTAPE 5 : PRÉDICTIONS ===
        print("\n4️⃣ Prédictions des modèles...")
        
        # LightGBM : prédit sur tout X_test
        y_proba_lgbm = lgbm_model.predict(X_test)
        y_pred_lgbm = (y_proba_lgbm >= lgbm_threshold).astype(int)
        
        # LSTM : prédit sur X_test_seq (taille réduite)
        y_proba_lstm = lstm_model.predict(X_test_seq, verbose=0).flatten()
        y_pred_lstm = (y_proba_lstm >= lstm_threshold).astype(int)
        
        # === ÉTAPE 6 : ALIGNEMENT DES DONNÉES ===
        print("\n5️⃣ Alignement des données pour comparaison équitable...")
        
        # PROBLÈME :
        # LightGBM a fait N prédictions (une par ligne de X_test)
        # LSTM a fait N-SEQUENCE_LENGTH prédictions (perd les premières lignes)
        #
        # SOLUTION :
        # On coupe les SEQUENCE_LENGTH premières prédictions de LightGBM
        # pour avoir exactement la même taille que LSTM
        
        start_index = SEQUENCE_LENGTH
        
        # Valeurs réelles communes (celles que LSTM peut prédire)
        y_test_common = y_test.values[start_index:]
        
        # Prédictions LightGBM alignées (on retire les premières)
        y_pred_lgbm_common = y_pred_lgbm[start_index:]
        y_proba_lgbm_common = y_proba_lgbm[start_index:]
        
        # y_test_seq, y_pred_lstm, y_proba_lstm sont déjà alignés
        # (créés par create_sequences)
        
        print(f"   ✅ Taille commune : {len(y_test_common):,} échantillons")
        print(f"   ℹ️ On compare sur les mêmes données pour être juste")
        
        # === ÉTAPE 7 : CALCUL DES MÉTRIQUES ===
        print("\n6️⃣ Calcul des métriques de performance...")
        
        # Métriques LightGBM (sur données alignées)
        metrics_lgbm = {
            'Accuracy': accuracy_score(y_test_common, y_pred_lgbm_common),
            'Precision': precision_score(y_test_common, y_pred_lgbm_common, zero_division=0),
            'Recall': recall_score(y_test_common, y_pred_lgbm_common, zero_division=0),
            'F1-Score': f1_score(y_test_common, y_pred_lgbm_common, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test_common, y_proba_lgbm_common)
        }
        
        # Métriques LSTM (déjà sur données alignées)
        metrics_lstm = {
            'Accuracy': accuracy_score(y_test_seq, y_pred_lstm),
            'Precision': precision_score(y_test_seq, y_pred_lstm, zero_division=0),
            'Recall': recall_score(y_test_seq, y_pred_lstm, zero_division=0),
            'F1-Score': f1_score(y_test_seq, y_pred_lstm, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test_seq, y_proba_lstm)
        }
        
        # Créer le tableau de comparaison
        comparison_df = generate_comparison_table(metrics_lgbm, metrics_lstm)
        
        # === AFFICHAGE DES RÉSULTATS ===
        print("\n" + "="*70)
        print("📊 RÉSULTATS DE LA COMPARAISON DES MODÈLES")
        print("="*70)
        print(tabulate(comparison_df.round(6), headers='keys', tablefmt='fancy_grid')) 
        
        # === ÉTAPE 8 : GRAPHIQUES ===
        if not args.no_plots:
            print("\n7️⃣ Génération des graphiques...")
            
            # Définir les chemins de sauvegarde (si --save-plots)
            save_cm = output_dir / "confusion_matrices.png" if args.save_plots else None
            save_roc = output_dir / "roc_curves.png" if args.save_plots else None
            save_pr = output_dir / "precision_recall_curves.png" if args.save_plots else None
            
            # Générer les 3 graphiques (sur données alignées)
            plot_confusion_matrices(y_test_common, y_pred_lgbm_common, y_pred_lstm, save_path=save_cm)
            plot_roc_curves(y_test_common, y_proba_lgbm_common, y_proba_lstm, save_path=save_roc)
            plot_precision_recall_curves(y_test_common, y_proba_lgbm_common, y_proba_lstm, save_path=save_pr)

        # === ÉTAPE 9 : RAPPORT TEXTE ===
        print("\n8️⃣ Génération du rapport final...")
        save_evaluation_report(comparison_df.round(6), output_dir)
        
        print("\n" + "="*70)
        print("✅ ÉVALUATION TERMINÉE AVEC SUCCÈS")
        print("="*70)
        print(f"\n📁 Résultats sauvegardés dans : {output_dir}")
        
    except Exception as e:
        print(f"\n❌ ERREUR LORS DE L'EXÉCUTION : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================================

if __name__ == "__main__":
    """
    Point d'entrée quand on exécute : python scripts/3_evaluate_models.py
    
    EXEMPLES D'UTILISATION :
    
    1. Afficher les graphiques interactifs :
       python scripts/3_evaluate_models.py
    
    2. Sans graphiques (mode rapide) :
       python scripts/3_evaluate_models.py --no-plots
    
    3. Sauvegarder les graphiques en PNG :
       python scripts/3_evaluate_models.py --save-plots
    
    SORTIE DU SCRIPT :
    - evaluation_results/confusion_matrices.png (si --save-plots)
    - evaluation_results/roc_curves.png (si --save-plots)
    - evaluation_results/precision_recall_curves.png (si --save-plots)
    - evaluation_results/evaluation_report_YYYYMMDD_HHMMSS.txt (toujours)
    
    DURÉE TYPIQUE : ~30 secondes
    """
    main()


# ============================================================================
# NOTES PÉDAGOGIQUES POUR DATA SCIENTIST JUNIOR
# ============================================================================

"""
📚 CONCEPTS CLÉS À RETENIR :

1. MÉTRIQUES D'ÉVALUATION
   -------------------------
   • Accuracy : % de prédictions correctes (simple mais trompeuse sur données déséquilibrées)
   • Precision : Parmi les prédictions positives, combien sont vraies ? (éviter fausses alertes)
   • Recall : Parmi les cas positifs réels, combien sont détectés ? (ne rien rater)
   • F1-Score : Moyenne harmonique de Precision et Recall (métrique d'équilibre)
   • ROC-AUC : Capacité à discriminer les classes (0.5 = aléatoire, 1.0 = parfait)

2. PROBLÈME D'ALIGNEMENT DES DONNÉES
   -----------------------------------
   LSTM perd SEQUENCE_LENGTH échantillons car il a besoin d'historique.
   
   Exemple avec SEQUENCE_LENGTH=12 :
   - Ligne 0 : Pas assez d'historique (besoin de 12 lignes avant)
   - Ligne 11 : Pas assez d'historique
   - Ligne 12 : OK ! (lignes 0-11 comme historique)
   
   Solution : On compare uniquement sur les lignes 12+ pour les 2 modèles.

3. CHOIX DE LA MÉTRIQUE PRINCIPALE
   ---------------------------------
   Pourquoi F1-Score ?
   - Notre dataset est déséquilibré (7% coupures)
   - Accuracy serait trompeuse (un modèle qui dit "jamais de coupure" aurait 93% d'accuracy !)
   - F1 pénalise les modèles qui négligent la classe minoritaire
   - C'est le standard pour classification déséquilibrée

4. INTERPRÉTATION DES GRAPHIQUES
   -------------------------------
   • Confusion Matrix : Visualise les 4 types d'erreurs (TN/FP/FN/TP)
   • ROC Curve : Trade-off entre Recall et Faux Positifs
   • PR Curve : Plus informative sur données déséquilibrées

5. COMPARAISON LIGHTGBM VS LSTM
   ------------------------------
   Typiquement sur ce projet :
   - LightGBM : Meilleur F1-Score, plus rapide, plus simple
   - LSTM : Capture mieux les dépendances temporelles longues, mais plus lourd
   
   LightGBM gagne souvent sur données tabulaires de taille moyenne (<100k lignes).

6. BONNES PRATIQUES
   -----------------
   ✅ Toujours comparer sur les mêmes données (alignement)
   ✅ Utiliser plusieurs métriques (pas seulement Accuracy)
   ✅ Visualiser les résultats (graphiques + tableaux)
   ✅ Sauvegarder un rapport texte (traçabilité)
   ✅ Tester avec différents seuils si besoin

7. COMMANDES UTILES
   -----------------
   # Évaluation complète avec graphiques
   python scripts/3_evaluate_models.py
   
   # Mode rapide sans graphiques
   python scripts/3_evaluate_models.py --no-plots
   
   # Sauvegarder les graphiques
   python scripts/3_evaluate_models.py --save-plots
   
   # Voir les résultats
   cat evaluation_results/evaluation_report_*.txt
"""