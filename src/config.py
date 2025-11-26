# Fichier : src/config.py
# Configuration centrale du projet Dakar Power Prediction
# =========================================================

import os
from pathlib import Path

# ====================================
# 1. CHEMINS DU PROJET
# ====================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Organisation des dossiers de données
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Dossier où je sauvegarde les modèles entraînés
MODELS_DIR = PROJECT_ROOT / "models"

# Fichiers de données spécifiques
RAW_DATA_FILE = DATA_RAW_DIR / "raw_data.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "processed_data.csv"

# ✅ CORRECTION : Tous les modèles utilisent .pkl (pas .joblib, pas .h5)
SCALER_FILE = MODELS_DIR / "scaler.pkl"           # StandardScaler pour normalisation
ENCODERS_FILE = MODELS_DIR / "encoders.pkl"       # LabelEncoder pour les quartiers
LGBM_MODEL_FILE = MODELS_DIR / "lgbm_model.pkl"   # ✅ Modèle LightGBM (.pkl)
LSTM_MODEL_FILE = MODELS_DIR / "lstm_model.h5"    # ✅ Modèle LSTM (.h5 - format Keras)

# Création automatique des dossiers
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ====================================
# 2. PARAMÈTRES DE GÉNÉRATION DE DONNÉES
# ====================================

START_DATE = '2024-01-01'
END_DATE = '2025-01-01'

QUARTIERS = [
    'Dakar-Plateau',
    'Parcelles Assainies',
    'Guediawaye',
    'Yoff',
    'Sicap-Liberté',
    'Mermoz-Sacré-Coeur'
]

QUARTIER_COORDS = {
    'Dakar-Plateau': {'lat': 14.667, 'lon': -17.433},
    'Yoff': {'lat': 14.767, 'lon': -17.483},
    'Mermoz-Sacré-Coeur': {'lat': 14.733, 'lon': -17.470},
    'Parcelles Assainies': {'lat': 14.760, 'lon': -17.440},
    'Guediawaye': {'lat': 14.783, 'lon': -17.417},
    'Sicap-Liberté': {'lat': 14.710, 'lon': -17.450}
}

MAP_CENTER = {'lat': 14.71, 'lon': -17.44}

PROBA_BASE_COUPURE = {
    'Dakar-Plateau': 0.02,
    'Parcelles Assainies': 0.08,
    'Guediawaye': 0.12,
    'Yoff': 0.04,
    'Sicap-Liberté': 0.06,
    'Mermoz-Sacré-Coeur': 0.032
}


# ====================================
# 3. PARAMÈTRES DE MODÉLISATION
# ====================================

FEATURE_COLUMNS = [
    'temp_celsius',
    'humidite_percent',
    'vitesse_vent',
    'conso_megawatt',
    'heure',
    'jour_semaine',
    'mois',
    'is_peak_hour',
    'quartier_encoded'
]

FEATURES_TO_SCALE = [
    'temp_celsius',
    'vitesse_vent',
    'conso_megawatt'
]

TARGET_COLUMN = 'coupure'

# --- Paramètres LSTM (Deep Learning) ---
SEQUENCE_LENGTH = 12
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# --- Paramètres LightGBM (Gradient Boosting) ---
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ====================================
# 4. PARAMÈTRES DE L'APPLICATION STREAMLIT
# ====================================

THRESHOLD_MODERATE = 0.15
THRESHOLD_HIGH = 0.30

HISTORICAL_HOURS = 24 * 7

DEFAULT_TEMP = 25.0
DEFAULT_HUMIDITE = 65.0
DEFAULT_VENT = 10.0
DEFAULT_CONSO = 800.0


# ====================================
# 5. BASE DE DONNÉES (FUTURE)
# ====================================

DATABASE_TYPE = 'mysql'

SQLITE_DB_FILE = DATA_DIR / "dakar_power.db"

MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'database': 'dakar_predictions',
    'user': 'root',
    'password': '',
    'charset': 'utf8mb4'
}

def get_db_connection_string():
    """
    Génère la chaîne de connexion SQLAlchemy selon le type de BDD.
    """
    if DATABASE_TYPE == 'sqlite':
        return f"sqlite:///{SQLITE_DB_FILE}"
    
    elif DATABASE_TYPE == 'mysql':
        user = MYSQL_CONFIG['user']
        password = MYSQL_CONFIG['password']
        host = MYSQL_CONFIG['host']
        port = MYSQL_CONFIG['port']
        database = MYSQL_CONFIG['database']
        charset = MYSQL_CONFIG['charset']
        
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset={charset}"
    
    else:
        raise ValueError(f"Type de base de données non supporté : {DATABASE_TYPE}")


# ====================================
# 6. COLONNES DU DATASET
# ====================================

COLUMN_NAMES = {
    'id': 'id_enregistrement',
    'datetime': 'date_heure',
    'quartier': 'quartier',
    'temperature': 'temp_celsius',
    'humidity': 'humidite_percent',
    'wind_speed': 'vitesse_vent',
    'consumption': 'conso_megawatt',
    'outage': 'coupure'
}


# ====================================
# 7. MESSAGES UTILISATEUR
# ====================================

MESSAGES = {
    'data_generated': "✅ Données générées avec succès !",
    'data_loaded': "✅ Données chargées avec succès !",
    'model_trained': "✅ Modèle entraîné avec succès !",
    'model_saved': "✅ Modèle sauvegardé avec succès !",
    'prediction_complete': "✅ Prédiction effectuée avec succès !",
    'error_data': "❌ Erreur lors du chargement des données.",
    'error_model': "❌ Erreur lors du chargement du modèle.",
}


# ====================================
# 8. FONCTION UTILITAIRE
# ====================================

def print_config():
    """Affiche un résumé de la configuration du projet."""
    print("="*50)
    print("CONFIGURATION DU PROJET DAKAR POWER PREDICTION")
    print("="*50)
    print(f"📁 Racine du projet : {PROJECT_ROOT}")
    print(f"📊 Fichier données brutes : {RAW_DATA_FILE}")
    print(f"🤖 Dossier modèles : {MODELS_DIR}")
    print(f"🏘️ Nombre de quartiers : {len(QUARTIERS)}")
    print(f"📅 Période de génération : {START_DATE} → {END_DATE}")
    print(f"🔢 Longueur séquence LSTM : {SEQUENCE_LENGTH} heures")
    print(f"⚠️ Seuils d'alerte : Modéré={THRESHOLD_MODERATE}, Élevé={THRESHOLD_HIGH}")
    print(f"🗄️ Type de base de données : {DATABASE_TYPE}")
    print(f"🎯 Features du modèle : {len(FEATURE_COLUMNS)} features")
    print(f"📦 Extensions modèles : LGBM=.pkl, LSTM=.h5")
    print("="*50)


if __name__ == "__main__":
    print_config()