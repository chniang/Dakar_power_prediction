# Fichier : streamlit_app/utils.py
# Fonctions utilitaires pour l'application Streamlit
# ====================================================
#
# OBJECTIF PRINCIPAL :
# Ce fichier contient toutes les fonctions réutilisables de l'interface Streamlit.
# Il sépare la logique métier (utils) de l'interface utilisateur (app.py).
#
# PRINCIPE DE CONCEPTION :
# "Separation of Concerns" - Chaque fonction a UNE responsabilité claire :
# - Chargement des modèles
# - Prédictions
# - Accès aux données
# - Validation des inputs
# - Formatage de l'affichage
#
# AVANTAGES DE CETTE ARCHITECTURE :
# ✅ Code réutilisable (fonctions appelées partout dans l'app)
# ✅ Tests faciles (chaque fonction testable indépendamment)
# ✅ Maintenance simple (bug ? chercher dans la fonction concernée)
# ✅ Performances (cache Streamlit pour éviter rechargements inutiles)
#
# STRUCTURE DU FICHIER :
# 1. Chargement des modèles (avec cache)
# 2. Fonctions de prédiction
# 3. Fonctions d'accès aux données
# 4. Fonctions d'affichage et validation
# 5. Utilitaires divers

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path
import sys
from datetime import datetime

# === CONFIGURATION DES CHEMINS ===
# Ajouter le dossier parent au path pour importer src/
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import (
    LGBM_MODEL_FILE, LSTM_MODEL_FILE, SCALER_FILE, ENCODERS_FILE,
    QUARTIER_COORDS, THRESHOLD_MODERATE, THRESHOLD_HIGH,
    SEQUENCE_LENGTH
)
from src.data_pipeline import DataPipeline
from src.database import DatabaseManager


# ============================================================================
# SECTION 1 : CHARGEMENT DES MODÈLES (AVEC CACHE)
# ============================================================================

@st.cache_resource
def load_models():
    """
    Charge les modèles ML, scaler et encodeur (UNE SEULE FOIS).
    
    CACHE STREAMLIT (@st.cache_resource) :
    Cette décoration est CRUCIALE pour les performances !
    
    Sans cache :
    - Les modèles se rechargent à CHAQUE interaction (slider, bouton)
    - Temps de chargement : ~5 secondes par interaction
    - Utilisateur frustré → application inutilisable
    
    Avec cache :
    - Chargement UNE FOIS au démarrage
    - Interactions instantanées ensuite
    - Les objets sont partagés entre toutes les sessions utilisateurs
    
    DIFFÉRENCE @cache_resource vs @cache_data :
    - @cache_resource : Pour objets NON-sérialisables (modèles, connexions DB)
    - @cache_data : Pour données sérialisables (DataFrames, listes, dict)
    
    QUAND LE CACHE EST VIDÉ :
    - Redémarrage de l'application
    - Modification du code de cette fonction
    - Bouton "Clear cache" dans l'interface Streamlit
    
    Returns:
        tuple: (lgbm_model, lgbm_threshold, lstm_model, lstm_threshold, 
                scaler, label_encoder) ou (None, ..., None) si erreur
    """
    try:
        # === CHARGEMENT LIGHTGBM ===
        # LightGBM est sauvegardé avec Joblib (format pickle optimisé)
        lgbm_data = joblib.load(LGBM_MODEL_FILE)
        lgbm_model = lgbm_data['model']      # Le modèle entraîné
        lgbm_threshold = lgbm_data['threshold']  # Seuil optimal (ex: 0.21)
        
        # === CHARGEMENT LSTM ===
        # LSTM est sauvegardé avec Keras (format HDF5)
        # compile=False : Pas besoin de recompiler (on fait juste des prédictions)
        lstm_model = tf.keras.models.load_model(LSTM_MODEL_FILE, compile=False)
        
        # Le seuil LSTM est dans un fichier texte séparé
        lstm_threshold_file = LSTM_MODEL_FILE.parent / "lstm_threshold.txt"
        with open(lstm_threshold_file, 'r') as f:
            lstm_threshold = float(f.read().strip())
        
        # === CHARGEMENT SCALER ===
        # StandardScaler pour normaliser les features (même échelle que l'entraînement)
        scaler = joblib.load(SCALER_FILE)
        
        # === CHARGEMENT ENCODEUR ===
        # LabelEncoder pour transformer les quartiers en nombres
        encoders = joblib.load(ENCODERS_FILE)
        label_encoder = encoders['quartier']
        
        return lgbm_model, lgbm_threshold, lstm_model, lstm_threshold, scaler, label_encoder
    
    except Exception as e:
        # Afficher l'erreur dans l'interface Streamlit (zone rouge)
        st.error(f"❌ Erreur lors du chargement des modèles : {e}")
        return None, None, None, None, None, None


@st.cache_resource
def get_database():
    """
    Initialise la connexion à la base de données (UNE SEULE FOIS).
    
    POURQUOI CACHER LA CONNEXION DB ?
    - Ouvrir/fermer une connexion DB à chaque requête est LENT
    - Le cache maintient la connexion ouverte pendant toute la session
    - Pool de connexions partagé entre utilisateurs
    
    GESTION D'ERREUR :
    Si la BD n'est pas disponible (fichier manquant, corruption), 
    l'application continue de fonctionner SANS historique.
    
    ALTERNATIVE SI BD INDISPONIBLE :
    - Prédictions en temps réel : ✅ Fonctionnent
    - Historique : ❌ Non disponible
    - Statistiques : ❌ Non disponibles
    
    Returns:
        DatabaseManager: Instance de connexion à la BD ou None si erreur
    """
    try:
        db = DatabaseManager()
        if db.connect():
            return db
        return None
    except Exception as e:
        # Warning (jaune) au lieu d'error (rouge) car l'app peut fonctionner sans BD
        st.warning(f"⚠️ Base de données non disponible : {e}")
        return None


# ============================================================================
# SECTION 2 : FONCTIONS DE PRÉDICTION
# ============================================================================

def make_prediction_single(input_data, quartier, lgbm_model, lgbm_threshold, 
                             lstm_model, lstm_threshold, scaler, label_encoder,
                             historical_data=None):
    """
    Effectue une prédiction pour une seule entrée utilisateur.
    
    WORKFLOW DE PRÉDICTION :
    1. Créer un DataFrame avec les données d'entrée
    2. Générer les features temporelles (heure, jour, mois, etc.)
    3. Encoder le quartier (texte → nombre)
    4. Normaliser les features (StandardScaler)
    5. Prédiction LightGBM (toujours disponible)
    6. Prédiction LSTM (si historique disponible)
    7. Calculer la probabilité moyenne
    8. Déterminer le statut de risque (Faible/Modéré/Élevé)
    
    POURQUOI DEUX MODÈLES ?
    - LightGBM : Rapide, précis, fonctionne TOUJOURS
    - LSTM : Capture les tendances temporelles, nécessite historique
    
    La moyenne des deux donne une prédiction plus robuste (ensemble learning).
    
    GESTION DES CAS LIMITES :
    - Quartier inconnu : Utilise la première classe connue (pas de crash)
    - Historique insuffisant : Utilise seulement LightGBM
    - Erreur LSTM : Utilise seulement LightGBM (graceful degradation)
    
    Args:
        input_data (dict): Données saisies par l'utilisateur
            {
                'temperature': float,   # °C
                'humidite': float,      # %
                'vent': float,          # km/h
                'consommation': float   # MW
            }
        quartier (str): Nom du quartier (ex: "Dakar-Plateau")
        lgbm_model: Modèle LightGBM chargé
        lgbm_threshold (float): Seuil de décision LightGBM (ex: 0.21)
        lstm_model: Modèle LSTM chargé
        lstm_threshold (float): Seuil de décision LSTM (ex: 0.50)
        scaler: StandardScaler pour normalisation
        label_encoder: LabelEncoder pour les quartiers
        historical_data (pd.DataFrame, optional): Données historiques pour LSTM
    
    Returns:
        dict: Résultats complets de prédiction
            {
                'proba_lgbm': float,        # Probabilité LightGBM (0.0-1.0)
                'pred_lgbm': int,           # Prédiction LightGBM (0 ou 1)
                'proba_lstm': float,        # Probabilité LSTM (0.0-1.0)
                'pred_lstm': int,           # Prédiction LSTM (0 ou 1)
                'proba_moyenne': float,     # Moyenne des probabilités
                'statut': str,              # "Risque Faible/Modéré/Élevé"
                'color': str,               # "green/orange/red"
                'emoji': str,               # "🟢/🟠/🔴"
                'seuil_lgbm': float,        # Seuil utilisé
                'seuil_lstm': float,        # Seuil utilisé
                'lstm_utilisable': bool     # LSTM a pu prédire ?
            }
    """
    
    # 1. Créer le DataFrame d'entrée
    df_input = pd.DataFrame([{
        'temp_celsius': input_data['temperature'],
        'humidite_percent': input_data['humidite'],
        'vitesse_vent': input_data['vent'],
        'conso_megawatt': input_data['consommation'],
        'date_heure': pd.Timestamp.now(),
        'quartier': quartier
    }])
    
    # 2. Créer les features temporelles
    pipeline = DataPipeline()
    df_input = pipeline.create_time_features(df_input)
    
    # 3. Encoder le quartier avec logique de secours (cohérence en production)
    try:
        if hasattr(label_encoder, 'classes_') and quartier in label_encoder.classes_:
            # Quartier connu
            df_input['quartier_encoded'] = label_encoder.transform([quartier])[0]
        else:
            # Quartier inconnu: utilise la première classe connue ou 0 par défaut
            if hasattr(label_encoder, 'classes_') and len(label_encoder.classes_) > 0:
                # Utilise la première classe pour éviter une erreur de LabelEncoder sur une nouvelle classe
                df_input['quartier_encoded'] = label_encoder.transform([label_encoder.classes_[0]])[0] 
            else:
                df_input['quartier_encoded'] = 0
    except Exception as e:
        df_input['quartier_encoded'] = 0
        print(f"⚠️ Erreur encodage quartier {quartier}: {e}. Utilisation valeur par défaut: 0")
    
    # 4. Préparer les features (9 colonnes)
    feature_cols = [
        'temp_celsius', 'humidite_percent', 'vitesse_vent', 'conso_megawatt',
        'heure', 'jour_semaine', 'mois', 'is_peak_hour', 'quartier_encoded'
    ]
    
    X_input = df_input[feature_cols].values
    
    # 5. Normaliser l'input (nécessaire pour LSTM et utilisé pour LGBM pour la cohérence du pipeline)
    X_scaled = scaler.transform(X_input)
    
    # 6. Prédiction LightGBM 
    # CORRECTION : Le modèle LightGBM (Booster) n'a pas de predict_proba().
    # On utilise predict(raw_score=True) pour obtenir le logit, puis on applique la sigmoïde.
    try:
        # Obtenir le logit (score brut)
        logit_lgbm = lgbm_model.predict(X_scaled, raw_score=True)[0]
        # Appliquer la fonction sigmoïde pour obtenir la probabilité P(Y=1)
        proba_lgbm = 1 / (1 + np.exp(-logit_lgbm))
    except Exception as e:
        # Logique de secours si raw_score n'est pas supporté ou autre erreur inattendue.
        print(f"⚠️ Erreur: Incapacité de calculer la probabilité LightGBM. Forcer proba à 0.05. Erreur: {e}")
        proba_lgbm = 0.05
        

    pred_lgbm = 1 if proba_lgbm >= lgbm_threshold else 0
    
    # 7. Prédiction LSTM
    proba_lstm = None
    pred_lstm = 0
    
    if historical_data is not None and len(historical_data) >= SEQUENCE_LENGTH - 1:
        try:
            # Préparer la séquence historique
            df_hist = historical_data.copy()
            df_hist = pipeline.create_time_features(df_hist)
            
            # Encoder les quartiers de l'historique (logique robuste)
            try:
                df_hist['quartier_encoded'] = df_hist['quartier'].apply(
                    lambda x: label_encoder.transform([x])[0] 
                    if hasattr(label_encoder, 'classes_') and x in label_encoder.classes_ 
                    else 0
                )
            except Exception as e:
                print(f"⚠️ Erreur encodage quartier historique: {e}. Utilisation valeur par défaut: 0")
                df_hist['quartier_encoded'] = 0 
            
            # Concaténer historique + nouvelle entrée
            df_sequence = pd.concat([
                df_hist.tail(SEQUENCE_LENGTH - 1)[feature_cols], 
                df_input[feature_cols]
            ])
            X_seq = df_sequence.values
            
            # Mettre à l'échelle la séquence (OBLIGATOIRE pour LSTM)
            X_seq_scaled = scaler.transform(X_seq)
            
            # Reshaper pour LSTM (samples, timesteps, features)
            X_seq_scaled = X_seq_scaled.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
            
            # Prédiction LSTM
            proba_lstm = lstm_model.predict(X_seq_scaled, verbose=0)[0][0]
            pred_lstm = 1 if proba_lstm >= lstm_threshold else 0
            
        except Exception as e:
            print(f"⚠️ Erreur prédiction LSTM: {e}. Utilisation uniquement de LightGBM.")
            proba_lstm = None
    
    # 8. Calculer la probabilité moyenne
    valid_probas = [proba_lgbm]
    if proba_lstm is not None:
        valid_probas.append(proba_lstm)
    
    # S'assurer qu'il y a des probabilités valides
    if not valid_probas:
        proba_moyenne = 0.0
    else:
        proba_moyenne = sum(valid_probas) / len(valid_probas)
    
    # Valeur d'affichage pour LSTM
    display_proba_lstm = proba_lstm if proba_lstm is not None else 0.0

    # 9. Déterminer le statut de risque
    if proba_moyenne >= THRESHOLD_HIGH:
        statut = "Risque Élevé"
        color = "red"
        emoji = "🔴"
    elif proba_moyenne >= THRESHOLD_MODERATE:
        statut = "Risque Modéré"
        color = "orange"
        emoji = "🟠"
    else:
        statut = "Risque Faible"
        color = "green"
        emoji = "🟢"
    
    return {
        'proba_lgbm': proba_lgbm,
        'pred_lgbm': pred_lgbm,
        'proba_lstm': display_proba_lstm,
        'pred_lstm': pred_lstm,
        'proba_moyenne': proba_moyenne,
        'statut': statut,
        'color': color,
        'emoji': emoji,
        'seuil_lgbm': lgbm_threshold,
        'seuil_lstm': lstm_threshold,
        'lstm_utilisable': proba_lstm is not None
    }


# ============================================================================
# SECTION 3 : FONCTIONS D'ACCÈS AUX DONNÉES
# ============================================================================

def get_historical_data(db, quartier=None, hours=168):
    """
    Récupère les données historiques depuis la BD
    
    USAGE TYPIQUE :
    - Afficher l'historique des coupures (graphiques)
    - Fournir des données pour LSTM (besoin de séquence temporelle)
    - Calculer des statistiques (taux de coupures récent)
    
    PARAMÈTRE hours=168 :
    168 heures = 7 jours (1 semaine d'historique par défaut)
    
    STRATÉGIE DE RÉCUPÉRATION :
    On demande hours * 2 enregistrements pour avoir une marge de sécurité.
    Pourquoi ? La BD peut avoir des trous (heures manquantes).
    
    Args:
        db (DatabaseManager): Instance de la BD
        quartier (str): Filtrer par quartier (optionnel)
        hours (int): Nombre d'heures à récupérer
        
    Returns:
        pd.DataFrame: Données historiques
    """
    if db is None:
        return pd.DataFrame()
    
    try:
        # Récupérer plus de points que nécessaire pour s'assurer d'avoir la séquence complète
        df = db.get_enregistrements(quartier=quartier, limit=hours * 2) 
        
        if not df.empty:
            df['date_heure'] = pd.to_datetime(df['date_heure'])
            df = df.sort_values('date_heure')
        
        return df
    except Exception as e:
        print(f"⚠️ Erreur récupération données historiques: {e}")
        return pd.DataFrame()


def get_statistics_by_quartier(db):
    """
    Récupère les statistiques par quartier
    
    MÉTRIQUES CALCULÉES :
    - total_enregistrements : Nombre d'observations
    - total_coupures : Nombre de coupures détectées
    - taux_coupure : % de coupures (0-100)
    - temp_moyenne : Température moyenne (°C)
    - conso_moyenne : Consommation moyenne (MW)
    
    USAGE :
    - Dashboard récapitulatif
    - Comparaison entre quartiers
    - Identification des zones à risque
    
    Args:
        db (DatabaseManager): Instance de la BD
        
    Returns:
        pd.DataFrame: Statistiques agrégées par quartier
    """
    if db is None:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT 
            quartier,
            COUNT(*) as total_enregistrements,
            SUM(coupure) as total_coupures,
            AVG(coupure) * 100 as taux_coupure,
            AVG(temp_celsius) as temp_moyenne,
            AVG(conso_megawatt) as conso_moyenne
        FROM enregistrements
        GROUP BY quartier
        ORDER BY taux_coupure DESC
        """
        
        df = pd.read_sql(query, db.engine)
        return df
    except Exception as e:
        print(f"⚠️ Erreur récupération statistiques: {e}")
        return pd.DataFrame()


# ============================================================================
# SECTION 4 : FONCTIONS D'AFFICHAGE ET VALIDATION
# ============================================================================

def format_percentage(value):
    """
    Formate un nombre en pourcentage
    
    Exemple:
        0.07234 → "7.23%"
        0.9 → "90.00%"
    
    Args:
        value (float): Nombre entre 0 et 1
    
    Returns:
        str: Pourcentage formaté
    """
    return f"{value * 100:.2f}%"


def get_risk_color(probability):
    """
    Retourne la couleur selon le niveau de risque
    
    MAPPING :
    - [0.7, 1.0] → "red" (Risque Élevé)
    - [0.3, 0.7[ → "orange" (Risque Modéré)
    - [0.0, 0.3[ → "green" (Risque Faible)
    
    Args:
        probability (float): Probabilité de coupure (0.0-1.0)
    
    Returns:
        str: "red", "orange" ou "green"
    """
    if probability >= THRESHOLD_HIGH:
        return "red"
    elif probability >= THRESHOLD_MODERATE:
        return "orange"
    else:
        return "green"


def display_metric_card(label, value, delta=None, help_text=None):
    """
    Affiche une métrique stylisée
    
    COMPOSANT STREAMLIT st.metric :
    Affiche une carte avec :
    - Label (titre)
    - Value (valeur principale)
    - Delta (variation, optionnel)
    - Help text (info-bulle, optionnel)
    
    Args:
        label (str): Titre de la métrique
        value: Valeur à afficher (peut être str, int, float)
        delta: Variation par rapport à une référence (optionnel)
        help_text (str): Texte d'aide au survol (optionnel)
    """
    st.metric(label=label, value=value, delta=delta, help=help_text)


def validate_input(temperature, humidite, vent, consommation):
    """
    Valide les entrées utilisateur
    
    RANGES DE VALIDATION :
    - Température : 15-40°C (climat de Dakar)
    - Humidité : 30-100% (physiquement possible)
    - Vent : 0-50 km/h (vents normaux à cycloniques)
    - Consommation : 200-1500 MW (capacité du réseau de Dakar)
    
    Args:
        temperature (float): Température en °C
        humidite (float): Humidité en %
        vent (float): Vitesse du vent en km/h
        consommation (float): Consommation en MW
    
    Returns:
        tuple: (is_valid, error_message)
    """
    errors = []
    
    if not (15 <= temperature <= 40):
        errors.append("❌ Température doit être entre 15°C et 40°C")
    
    if not (30 <= humidite <= 100):
        errors.append("❌ Humidité doit être entre 30% et 100%")
    
    if not (0 <= vent <= 50):
        errors.append("❌ Vitesse du vent doit être entre 0 et 50 km/h")
    
    if not (200 <= consommation <= 1500):
        errors.append("❌ Consommation doit être entre 200 et 1500 MW")
    
    if errors:
        return False, "\n".join(errors)
    
    return True, ""


# ============================================================================
# SECTION 5 : UTILITAIRES DIVERS
# ============================================================================

def get_quartier_coords():
    """
    Retourne les coordonnées des quartiers
    
    SOURCE : config.py → QUARTIER_COORDS
    
    FORMAT :
    {
        'Dakar-Plateau': {'lat': 14.6937, 'lon': -17.4441},
        'Guédiawaye': {'lat': 14.7692, 'lon': -17.3862},
        ...
    }
    
    USAGE :
    Afficher les quartiers sur une carte Streamlit
    
    Returns:
        dict: Coordonnées GPS par quartier
    """
    return QUARTIER_COORDS


def get_quartier_list():
    """
    Retourne la liste des quartiers
    
    USAGE :
    - Populate un selectbox Streamlit
    - Valider un nom de quartier
    
    Returns:
        list: Liste des noms de quartiers
    """
    return list(QUARTIER_COORDS.keys())


def save_prediction_to_db(db, prediction_data):
    """
    Sauvegarde une prédiction dans la BD
    
    POURQUOI SAUVEGARDER LES PRÉDICTIONS ?
    - Traçabilité : Historique des prédictions faites
    - Analyse : Comparer prédictions vs réalité
    - Monitoring : Détecter les dérives du modèle
    - Audit : Qui a prédit quoi et quand ?
    
    DONNÉES SAUVEGARDÉES :
    - Date/heure de la prédiction
    - Quartier concerné
    - Features d'entrée (temp, humidité, vent, conso)
    - Probabilités prédites (LightGBM, LSTM, moyenne)
    - Statut de risque (Faible/Modéré/Élevé)
    
    Args:
        db (DatabaseManager): Instance de la BD
        prediction_data (dict): Données de prédiction
        
    Returns:
        int: ID de la prédiction (ou None si erreur)
    """
    if db is None:
        return None
    
    try:
        pred_id = db.insert_prediction(prediction_data)
        return pred_id
    except Exception as e:
        print(f"⚠️ Erreur sauvegarde prédiction en BD: {e}")
        return None


# ============================================================================
# NOTES PÉDAGOGIQUES POUR DATA SCIENTIST JUNIOR
# ============================================================================

"""
📚 CONCEPTS CLÉS À RETENIR :

1. ARCHITECTURE MODULAIRE (SEPARATION OF CONCERNS)
   ------------------------------------------------
   Ce fichier utils.py sépare la LOGIQUE MÉTIER de l'INTERFACE UTILISATEUR.
   
   Principe :
   ❌ MAUVAIS : Tout dans app.py (1000+ lignes, illisible)
   ✅ BON : Logique dans utils.py, UI dans app.py
   
   Avantages :
   - Code réutilisable (fonctions appelées partout)
   - Tests faciles (chaque fonction testable indépendamment)
   - Maintenance simple (1 bug = 1 fonction à corriger)
   - Collaboration facilitée (plusieurs développeurs)

2. CACHE STREAMLIT (@st.cache_resource et @st.cache_data)
   -------------------------------------------------------
   Le cache est ESSENTIEL pour les performances de Streamlit.
   
   Sans cache : Chaque interaction (clic, slider) RECHARGE TOUT
   Avec cache : Chargement UNE FOIS, puis réutilisation
   
   Deux types de cache :
   
   @st.cache_resource → Pour objets NON-sérialisables
   - Modèles ML (LightGBM, LSTM)
   - Connexions BD
   - Sessions réseau
   
   @st.cache_data → Pour données sérialisables
   - DataFrames
   - Listes, dictionnaires
   - Résultats de calculs

3. GESTION ROBUSTE DES ERREURS (GRACEFUL DEGRADATION)
   ---------------------------------------------------
   Une bonne application ne crash JAMAIS pour l'utilisateur.
   
   Principe : Si quelque chose échoue, l'app continue avec fonctionnalités réduites.
   
   Exemples dans ce fichier :
   - BD inaccessible ? → Prédictions temps réel fonctionnent toujours
   - LSTM échoue ? → Utilise seulement LightGBM
   - Quartier inconnu ? → Utilise une valeur par défaut

4. PRÉDICTIONS ENSEMBLE (LIGHTGBM + LSTM)
   ----------------------------------------
   On utilise DEUX modèles pour plus de robustesse.
   
   LightGBM :
   - Rapide (millisecondes)
   - Fonctionne toujours (pas besoin d'historique)
   - Excellent sur données tabulaires
   
   LSTM :
   - Capture les tendances temporelles
   - Nécessite historique (12 heures)
   - Plus lent (quelques secondes)
   
   Prédiction finale = MOYENNE des deux

5. NORMALISATION DES FEATURES (STANDARDSCALER)
   --------------------------------------------
   CRITIQUE : Les features doivent avoir la même échelle qu'à l'entraînement.
   
   StandardScaler transforme : X_scaled = (X - mean) / std
   
   ⚠️ ATTENTION : Utiliser le MÊME scaler qu'à l'entraînement !
   - scaler.fit() → À l'entraînement (calcule mean/std)
   - scaler.transform() → En production (applique mean/std)
   
   Ne JAMAIS appeler fit() en production !

6. VALIDATION DES INPUTS UTILISATEUR
   -----------------------------------
   JAMAIS faire confiance aux entrées utilisateur.
   
   validate_input() vérifie les ranges AVANT prédiction.
   
   Bonnes pratiques :
   ✅ Valider côté client (Streamlit sliders avec min/max)
   ✅ Valider côté serveur (validate_input())
   ✅ Afficher des messages d'erreur clairs

7. COMMANDES UTILES
   -----------------
   # Lancer l'application Streamlit
   streamlit run streamlit_app/app.py
   
   # Avec debug (auto-reload)
   streamlit run streamlit_app/app.py --server.runOnSave true
   
   # Tester une fonction utils
   python -c "from streamlit_app.utils import load_models; print(load_models())"
   
   # Clear cache manuellement
   # Dans l'app : Menu (☰) > Clear cache

8. ERREURS COURANTES ET SOLUTIONS
   --------------------------------
   ❌ "Session state has no attribute X"
   ✅ Initialiser dans app.py : if 'X' not in st.session_state: st.session_state.X = default
   
   ❌ "Model file not found"
   ✅ Vérifier que les modèles sont entraînés (python scripts/2_train_models.py)
   
   ❌ "Scaler expects X features but got Y"
   ✅ Vérifier que feature_cols a le bon ordre et nombre de colonnes
   
   ❌ "LabelEncoder: classes_ not found"
   ✅ Vérifier que l'encodeur est bien sauvegardé après l'entraînement
   
   ❌ Page blanche / app ne démarre pas
   ✅ Vérifier les imports (pip install -r requirements.txt)
   
   ❌ Cache ne se vide pas
   ✅ Redémarrer l'app (Ctrl+C puis relancer)

9. GESTION DES SÉQUENCES TEMPORELLES (LSTM)
   -----------------------------------------
   LSTM nécessite une séquence de SEQUENCE_LENGTH observations (ex: 12 heures).
   
   Format d'entrée LSTM : (samples, timesteps, features)
   - samples = 1 (une prédiction à la fois)
   - timesteps = 12 (12 heures d'historique)
   - features = 9 (9 colonnes)
   
   Shape finale : (1, 12, 9)
   
   Construction de la séquence :
   1. Récupérer les 11 dernières heures de l'historique
   2. Ajouter l'observation actuelle (1 heure)
   3. Total : 12 heures
   4. Normaliser TOUTE la séquence
   5. Reshaper pour LSTM
   
   Si historique insuffisant (<11 heures) :
   → LSTM non utilisable, utilise seulement LightGBM

10. PROBABILITÉS VS PRÉDICTIONS BINAIRES
    -------------------------------------
    Les modèles retournent des PROBABILITÉS (0.0-1.0), pas des 0/1.
    
    Probabilité : "Il y a 73% de chance de coupure"
    Prédiction : "Oui, coupure" (si proba >= seuil)
    
    Conversion :
    ```python
    proba = 0.73
    seuil = 0.21  # Seuil optimal trouvé à l'entraînement
    pred = 1 if proba >= seuil else 0
    ```
    
    Pourquoi afficher les probabilités ?
    - Plus informatif ("73%" > "Oui")
    - Permet à l'utilisateur de juger
    - Utile pour fixer des seuils personnalisés

11. STATUT DE RISQUE (FAIBLE/MODÉRÉ/ÉLEVÉ)
    ---------------------------------------
    On transforme les probabilités en statuts compréhensibles.
    
    Mapping (exemple avec seuils config.py) :
    - [0.0, 0.3[ → 🟢 Risque Faible (green)
    - [0.3, 0.7[ → 🟠 Risque Modéré (orange)
    - [0.7, 1.0] → 🔴 Risque Élevé (red)
    
    Pourquoi ?
    - Utilisateurs non techniques préfèrent "Risque Élevé" à "0.82"
    - Couleurs facilitent la lecture (rouge = danger)
    - Emojis augmentent l'attention
    
    Ces seuils sont configurables dans config.py :
    - THRESHOLD_MODERATE = 0.3
    - THRESHOLD_HIGH = 0.7

12. BONNES PRATIQUES - FONCTIONS PURES
    ------------------------------------
    Les fonctions de utils.py sont "pures" quand possible.
    
    Fonction pure :
    - Pas d'effets de bord
    - Même input → Même output (déterministe)
    - Pas de modification d'état global
    
    Exemple :
    ```python
    # ✅ PURE
    def format_percentage(value):
        return f"{value * 100:.2f}%"
    
    # ❌ IMPURE (modifie une variable globale)
    counter = 0
    def format_percentage(value):
        global counter
        counter += 1  # Effet de bord !
        return f"{value * 100:.2f}%"
    ```
    
    Avantages des fonctions pures :
    - Testables facilement
    - Pas de surprises
    - Parallélisables
    - Cachables (memoization)

13. STRUCTURE D'UN BON FICHIER UTILS
    ----------------------------------
    Organisation logique par SECTIONS :
    
    1. Imports
    2. Chargement des ressources (modèles, BD)
    3. Logique métier principale (prédictions)
    4. Accès aux données
    5. Affichage et validation
    6. Utilitaires divers
    
    Chaque section = une responsabilité.

14. DEBUGGING STREAMLIT
    --------------------
    Outils utiles :
    
    # Afficher des variables en debug
    st.write("Debug:", variable)
    
    # Afficher un DataFrame
    st.dataframe(df)
    
    # Afficher un objet JSON
    st.json(dict_object)
    
    # Logs dans la console
    print(f"⚠️ Debug: {variable}")
    
    # Exception avec stack trace
    st.exception(exception_object)
    
    # Progress bar pour opérations longues
    with st.spinner('Calcul en cours...'):
        result = long_operation()

15. OPTIMISATIONS POSSIBLES
    ------------------------
    Ce fichier est déjà optimisé, mais possibilités d'amélioration :
    
    - Batch predictions (prédire plusieurs quartiers en une fois)
    - Async DB queries (requêtes parallèles)
    - Compression des modèles (quantization)
    - CDN pour assets statiques
    - Redis pour cache distribué
    - Monitoring (Prometheus, Grafana)
    - A/B testing (tester différents seuils)

16. RÉCAPITULATIF DES FONCTIONS
    ----------------------------
    CHARGEMENT :
    - load_models() : Charge LightGBM, LSTM, scaler, encodeur (avec cache)
    - get_database() : Initialise la connexion BD (avec cache)
    
    PRÉDICTIONS :
    - make_prediction_single() : Fait une prédiction complète (LightGBM + LSTM)
    
    DONNÉES :
    - get_historical_data() : Récupère l'historique depuis la BD
    - get_statistics_by_quartier() : Calcule les stats agrégées
    
    AFFICHAGE :
    - format_percentage() : Formate 0.07 → "7.00%"
    - get_risk_color() : Probabilité → "green/orange/red"
    - display_metric_card() : Affiche une métrique Streamlit
    - validate_input() : Valide les entrées utilisateur
    
    UTILITAIRES :
    - get_quartier_coords() : Retourne les coordonnées GPS
    - get_quartier_list() : Liste des noms de quartiers
    - save_prediction_to_db() : Sauvegarde une prédiction

17. WORKFLOW TYPIQUE D'UNE PRÉDICTION
    -----------------------------------
    1. Utilisateur saisit : température, humidité, vent, consommation, quartier
    2. validate_input() vérifie les ranges
    3. make_prediction_single() est appelée :
       a. Création du DataFrame
       b. Génération des features temporelles
       c. Encodage du quartier
       d. Normalisation (StandardScaler)
       e. Prédiction LightGBM
       f. Prédiction LSTM (si historique disponible)
       g. Calcul de la probabilité moyenne
       h. Détermination du statut de risque
    4. Résultats affichés dans l'interface Streamlit
    5. (Optionnel) save_prediction_to_db() sauvegarde dans la BD

18. DÉPENDANCES CRITIQUES
    ----------------------
    Ce fichier dépend de :
    
    MODULES INTERNES :
    - src/config.py : Constantes (chemins, seuils, coordonnées)
    - src/data_pipeline.py : DataPipeline.create_time_features()
    - src/database.py : DatabaseManager
    
    BIBLIOTHÈQUES EXTERNES :
    - streamlit : Framework d'interface
    - pandas : Manipulation de données
    - numpy : Calculs numériques
    - joblib : Chargement modèles/scaler/encodeur
    - tensorflow : Chargement LSTM
    
    FICHIERS REQUIS (générés par l'entraînement) :
    - models/lgbm_model.joblib : Modèle LightGBM
    - models/lstm_model.h5 : Modèle LSTM
    - models/lstm_threshold.txt : Seuil LSTM
    - models/scaler.joblib : StandardScaler
    - models/encoders.joblib : LabelEncoder
    
    Si un fichier manque → Erreur au chargement → Affichage dans Streamlit

19. TESTS UNITAIRES POSSIBLES
    --------------------------
    Exemples de tests à écrire pour ce module :
    
    ```python
    def test_format_percentage():
        assert format_percentage(0.07) == "7.00%"
        assert format_percentage(1.0) == "100.00%"
    
    def test_validate_input():
        # Valid
        is_valid, _ = validate_input(25, 60, 10, 500)
        assert is_valid == True
        
        # Invalid temperature
        is_valid, msg = validate_input(50, 60, 10, 500)
        assert is_valid == False
        assert "Température" in msg
    
    def test_get_risk_color():
        assert get_risk_color(0.1) == "green"
        assert get_risk_color(0.5) == "orange"
        assert get_risk_color(0.9) == "red"
    
    def test_get_quartier_list():
        quartiers = get_quartier_list()
        assert len(quartiers) > 0
        assert "Dakar-Plateau" in quartiers
    ```

20. MAINTENANCE ET ÉVOLUTION
    -------------------------
    Ce fichier est stable mais peut évoluer :
    
    AJOUTS FUTURS POSSIBLES :
    - Nouvelles métriques d'affichage
    - Support de nouveaux modèles (XGBoost, Random Forest)
    - Prédictions batch (plusieurs quartiers simultanément)
    - Export des résultats (PDF, Excel)
    - Notifications par email/SMS
    - Intégration API externe (météo en temps réel)
    
    RÈGLES DE MAINTENANCE :
    - Une fonction = une responsabilité (Single Responsibility Principle)
    - Toujours documenter les nouvelles fonctions
    - Ajouter des tests unitaires
    - Maintenir la cohérence du style de code
    - Versionner les changements (Git)
    
    SIGNAUX D'ALERTE :
    - Fonction > 50 lignes → Décomposer
    - Duplication de code → Factoriser
    - Trop de try/except imbriqués → Simplifier
    - Import circulaire → Revoir l'architecture
"""