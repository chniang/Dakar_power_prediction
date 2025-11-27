# Fichier : src/config.py
# Configuration centrale avec Supabase (PostgreSQL)
# =========================================================

from pathlib import Path
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Chemins du projet
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'

# Créer les dossiers s'ils n'existent pas
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Fichiers de données
METEO_FILE = RAW_DATA_DIR / 'data_meteo_dakar.csv'
POWER_FILE = RAW_DATA_DIR / 'data_energy_dakar.csv'
PROCESSED_FILE = PROCESSED_DATA_DIR / 'dakar_power_ml_ready.csv'

# Alias pour compatibilité avec data_pipeline.py
RAW_DATA_FILE = RAW_DATA_DIR / 'dakar_power_raw.csv'
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / 'dakar_power_ml_ready.csv'

# Modèles
LGBM_MODEL_FILE = MODELS_DIR / "lgbm_model.pkl"
LSTM_MODEL_FILE = MODELS_DIR / "lstm_model.h5"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
ENCODER_FILE = MODELS_DIR / "encoders.pkl"
ENCODERS_FILE = MODELS_DIR / "encoders.pkl"  # Alias pour compatibilité

# ==========================================
# CONFIGURATION DATABASE (SUPABASE/POSTGRESQL)
# ==========================================

# Pour Streamlit Cloud (utilise secrets.toml)
try:
    import streamlit as st
    DB_CONFIG = {
        'host': st.secrets["database"]["host"],
        'port': st.secrets["database"]["port"],
        'database': st.secrets["database"]["database"],
        'user': st.secrets["database"]["user"],
        'password': st.secrets["database"]["password"]
    }
except:
    # Pour développement local (utilise .env)
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'postgres'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }

# SQLAlchemy connection string pour PostgreSQL
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# ==========================================
# FEATURES & QUARTIERS
# ==========================================

FEATURE_COLS = [
    'temperature', 'humidite', 'vitesse_vent', 'consommation_electrique',
    'heure', 'jour_semaine', 'mois', 'saison', 'est_weekend'
]

# Alias pour compatibilité avec data_pipeline.py
FEATURE_COLUMNS = FEATURE_COLS
FEATURES_TO_SCALE = ['temperature', 'humidite', 'vitesse_vent', 'consommation_electrique']
TARGET_COLUMN = 'coupure'

QUARTIERS_DAKAR = [
    'Dakar-Plateau', 'Yoff', 'Mermoz-Sacré-Coeur',
    'Parcelles Assainies', 'Guediawaye', 'Sicap-Liberté'
]

# ==========================================
# COORDONNÉES DES QUARTIERS
# ==========================================

QUARTIER_COORDS = {
    'Dakar-Plateau': {'lat': 14.6937, 'lon': -17.4441},
    'Yoff': {'lat': 14.7500, 'lon': -17.4900},
    'Mermoz-Sacré-Coeur': {'lat': 14.7200, 'lon': -17.4600},
    'Parcelles Assainies': {'lat': 14.7800, 'lon': -17.4400},
    'Guediawaye': {'lat': 14.7700, 'lon': -17.4200},
    'Sicap-Liberté': {'lat': 14.7100, 'lon': -17.4500}
}

# ==========================================
# SEUILS DE RISQUE
# ==========================================

SEUILS_RISQUE = {
    'faible': 30,
    'moyen': 50,
    'eleve': 70
}

# Seuils (alias pour compatibilité avec utils.py)
THRESHOLD_MODERATE = 30
THRESHOLD_HIGH = 70

# ==========================================
# PARAMÈTRES LSTM
# ==========================================

SEQUENCE_LENGTH = 12  # Longueur des séquences pour LSTM


# ==========================================
# PARAMÈTRES D'ENTRAÎNEMENT
# ==========================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2

# Hyperparamètres LightGBM
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
    'random_state': RANDOM_STATE
}

# Hyperparamètres LSTM
LSTM_PARAMS = {
    'units_1': 100,
    'units_2': 50,
    'dropout': 0.4,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001
}

print("✅ Configuration chargée (Supabase/PostgreSQL)")