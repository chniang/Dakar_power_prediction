# Fichier : src/config.py
# Configuration centrale du projet Dakar Power Prediction
# =========================================================

import os
from pathlib import Path

# ====================================
# CHEMINS DU PROJET
# ====================================

# Racine du projet
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Dossiers de données
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Dossier des modèles
MODELS_DIR = PROJECT_ROOT / "models"

# Fichiers de données
RAW_DATA_FILE = DATA_RAW_DIR / "raw_data.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "processed_data.csv"

# Fichiers des modèles
SCALER_FILE = MODELS_DIR / "scaler.pkl"
ENCODERS_FILE = MODELS_DIR / "encoders.pkl"
LGBM_MODEL_FILE = MODELS_DIR / "lgbm_model.pkl"
LSTM_MODEL_FILE = MODELS_DIR / "lstm_model.keras"

# Créer les dossiers s'ils n'existent pas
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ====================================
# PARAMÈTRES DE GÉNÉRATION DE DONNÉES
# ====================================

# Période de génération
START_DATE = '2024-01-01'
END_DATE = '2025-01-01'

# Quartiers de Dakar
QUARTIERS = [
    'Dakar-Plateau',
    'Parcelles Assainies',
    'Guediawaye',
    'Yoff',
    'Sicap-Liberté',
    'Mermoz-Sacré-Coeur'
]

# Coordonnées GPS des quartiers
QUARTIER_COORDS = {
    'Dakar-Plateau': {'lat': 14.667, 'lon': -17.433},
    'Yoff': {'lat': 14.767, 'lon': -17.483},
    'Mermoz-Sacré-Coeur': {'lat': 14.733, 'lon': -17.470},
    'Parcelles Assainies': {'lat': 14.760, 'lon': -17.440},
    'Guediawaye': {'lat': 14.783, 'lon': -17.417},
    'Sicap-Liberté': {'lat': 14.710, 'lon': -17.450}
}

# Centre de la carte (Dakar)
MAP_CENTER = {'lat': 14.71, 'lon': -17.44}

# Probabilités de base de coupure par quartier
PROBA_BASE_COUPURE = {
    'Dakar-Plateau': 0.02,
    'Parcelles Assainies': 0.08,
    'Guediawaye': 0.12,  # Zone à risque élevé
    'Yoff': 0.04,
    'Sicap-Liberté': 0.06,
    'Mermoz-Sacré-Coeur': 0.032
}


# ====================================
# PARAMÈTRES DE MODÉLISATION
# ====================================

# Features pour le modèle
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

# Colonnes à normaliser
FEATURES_TO_SCALE = [
    'temp_celsius',
    'vitesse_vent',
    'conso_megawatt'
]

# Target
TARGET_COLUMN = 'coupure'

# Paramètres LSTM
SEQUENCE_LENGTH = 12  # 12 heures de séquence
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# Paramètres LightGBM
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

# Split train/test
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ====================================
# PARAMÈTRES DE L'APPLICATION
# ====================================

# Seuil d'alerte
THRESHOLD_MODERATE = 0.15  # Risque modéré
THRESHOLD_HIGH = 0.30      # Risque élevé

# Historique à afficher (en heures)
HISTORICAL_HOURS = 24 * 7  # 7 jours

# Valeurs par défaut des sliders
DEFAULT_TEMP = 25.0
DEFAULT_HUMIDITE = 65.0
DEFAULT_VENT = 10.0
DEFAULT_CONSO = 800.0


# ====================================
# BASE DE DONNÉES (FUTURE)
# ====================================

# Type de base de données
DATABASE_TYPE = 'sqlite'  # 'sqlite' ou 'mysql'

# SQLite (pour développement local)
SQLITE_DB_FILE = DATA_DIR / "dakar_power.db"

# MySQL (pour production)
MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'database': 'dakar_power',
    'user': 'root',
    'password': 'your_password',
    'charset': 'utf8mb4'
}

# Connection string SQLAlchemy
def get_db_connection_string():
    """Retourne la chaîne de connexion selon le type de BD"""
    if DATABASE_TYPE == 'sqlite':
        return f"sqlite:///{SQLITE_DB_FILE}"
    elif DATABASE_TYPE == 'mysql':
        return f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}?charset={MYSQL_CONFIG['charset']}"
    else:
        raise ValueError(f"Type de base de données non supporté : {DATABASE_TYPE}")


# ====================================
# COLONNES DU DATASET
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
# MESSAGES
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
# FONCTION UTILITAIRE
# ====================================

def print_config():
    """Affiche la configuration du projet"""
    print("="*50)
    print("CONFIGURATION DU PROJET")
    print("="*50)
    print(f"📁 Racine du projet : {PROJECT_ROOT}")
    print(f"📊 Fichier données brutes : {RAW_DATA_FILE}")
    print(f"🤖 Dossier modèles : {MODELS_DIR}")
    print(f"🏘️ Nombre de quartiers : {len(QUARTIERS)}")
    print(f"📅 Période : {START_DATE} → {END_DATE}")
    print(f"🔢 Longueur séquence LSTM : {SEQUENCE_LENGTH}h")
    print(f"⚠️ Seuils : Modéré={THRESHOLD_MODERATE}, Élevé={THRESHOLD_HIGH}")
    print("="*50)


if __name__ == "__main__":
    print_config()