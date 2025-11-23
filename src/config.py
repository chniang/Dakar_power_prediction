# Fichier : src/config.py
# Configuration centrale du projet Dakar Power Prediction
# =========================================================
#
# Ce fichier regroupe TOUTES les configurations du projet en un seul endroit.
# Pourquoi ? Pour faciliter la maintenance : si je veux changer un paramètre,
# je viens ici au lieu de chercher dans 10 fichiers différents.
#
# Organisation :
# 1. Chemins des dossiers et fichiers
# 2. Paramètres de génération des données
# 3. Paramètres des modèles ML
# 4. Paramètres de l'application Streamlit
# 5. Configuration base de données

import os
from pathlib import Path

# ====================================
# 1. CHEMINS DU PROJET
# ====================================
# Je définis tous les chemins ici pour éviter les erreurs de chemin relatif.
# Path(__file__) = ce fichier actuel (config.py)
# .parent = dossier parent (src/)
# .parent.parent = racine du projet (dakar_power_prediction/)

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Organisation des dossiers de données
# data/raw/ → données brutes générées
# data/processed/ → données après prétraitement
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Dossier où je sauvegarde les modèles entraînés
MODELS_DIR = PROJECT_ROOT / "models"

# Fichiers de données spécifiques
RAW_DATA_FILE = DATA_RAW_DIR / "raw_data.csv"           # 52,704 lignes générées
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "processed_data.csv"  # Avec features engineering

# Fichiers des modèles et transformers
SCALER_FILE = MODELS_DIR / "scaler.pkl"           # StandardScaler pour normalisation
ENCODERS_FILE = MODELS_DIR / "encoders.pkl"       # LabelEncoder pour les quartiers
LGBM_MODEL_FILE = MODELS_DIR / "lgbm_model.pkl"   # Modèle LightGBM entraîné
LSTM_MODEL_FILE = MODELS_DIR / "lstm_model.keras" # Modèle LSTM (TensorFlow)

# Création automatique des dossiers s'ils n'existent pas encore
# Ça évite les erreurs "FileNotFoundError" lors de la première exécution
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ====================================
# 2. PARAMÈTRES DE GÉNÉRATION DE DONNÉES
# ====================================
# Ces paramètres contrôlent la génération des 52,704 enregistrements synthétiques

# Période d'un an complet (8,784 heures)
START_DATE = '2024-01-01'
END_DATE = '2025-01-01'

# Les 6 quartiers de Dakar que j'analyse
# Ordre alphabétique pour faciliter la lecture
QUARTIERS = [
    'Dakar-Plateau',           # Centre administratif, stable
    'Parcelles Assainies',     # Banlieue nord
    'Guediawaye',              # Zone la plus fragile (coupures fréquentes)
    'Yoff',                    # Zone aéroportuaire
    'Sicap-Liberté',           # Zone résidentielle
    'Mermoz-Sacré-Coeur'       # Zone mixte résidentiel/commercial
]

# Coordonnées GPS réelles pour afficher la carte dans Streamlit
# Source : Google Maps
QUARTIER_COORDS = {
    'Dakar-Plateau': {'lat': 14.667, 'lon': -17.433},
    'Yoff': {'lat': 14.767, 'lon': -17.483},
    'Mermoz-Sacré-Coeur': {'lat': 14.733, 'lon': -17.470},
    'Parcelles Assainies': {'lat': 14.760, 'lon': -17.440},
    'Guediawaye': {'lat': 14.783, 'lon': -17.417},
    'Sicap-Liberté': {'lat': 14.710, 'lon': -17.450}
}

# Centre de la carte pour Streamlit (moyenne approximative)
MAP_CENTER = {'lat': 14.71, 'lon': -17.44}

# Probabilités de base de coupure par quartier (basées sur observations terrain)
# Guediawaye a 6× plus de coupures que Dakar-Plateau
# Ces valeurs servent de point de départ, ensuite on ajoute l'effet météo/consommation
PROBA_BASE_COUPURE = {
    'Dakar-Plateau': 0.02,          # 2% (infrastructure moderne)
    'Parcelles Assainies': 0.08,    # 8% (réseau surchargé)
    'Guediawaye': 0.12,             # 12% (zone la plus fragile)
    'Yoff': 0.04,                   # 4% (proche aéroport, prioritaire)
    'Sicap-Liberté': 0.06,          # 6% (densité moyenne)
    'Mermoz-Sacré-Coeur': 0.032     # 3.2% (zone stable)
}


# ====================================
# 3. PARAMÈTRES DE MODÉLISATION
# ====================================

# Les 9 features que j'utilise pour entraîner les modèles
# Ordre important : doit correspondre à l'ordre dans le DataFrame
FEATURE_COLUMNS = [
    'temp_celsius',        # Température en °C
    'humidite_percent',    # Humidité relative en %
    'vitesse_vent',        # Vent en km/h
    'conso_megawatt',      # Consommation électrique en MW
    'heure',               # Heure de la journée (0-23)
    'jour_semaine',        # Jour de la semaine (0=Lundi, 6=Dimanche)
    'mois',                # Mois de l'année (1-12)
    'is_peak_hour',        # Binaire : 1=heure de pointe, 0=sinon
    'quartier_encoded'     # Quartier encodé en nombre (0-5)
]

# Features à normaliser avec StandardScaler
# Je normalise uniquement les features continues (pas heure, jour, mois)
# Pourquoi ? StandardScaler transforme : (X - moyenne) / écart-type
FEATURES_TO_SCALE = [
    'temp_celsius',        # Varie de 15-40°C
    'vitesse_vent',        # Varie de 0-50 km/h
    'conso_megawatt'       # Varie de 200-1500 MW
]
# Note : J'ai retiré 'humidite_percent' car elle causait des problèmes de corrélation

# Variable cible (ce qu'on veut prédire)
TARGET_COLUMN = 'coupure'  # 0 = pas de coupure, 1 = coupure

# --- Paramètres LSTM (Deep Learning) ---
SEQUENCE_LENGTH = 12   # Le LSTM regarde les 12 dernières heures pour prédire
LSTM_UNITS = 64        # Nombre de neurones dans les couches LSTM
LSTM_DROPOUT = 0.2     # Dropout pour éviter l'overfitting (20% neurones désactivés)
LSTM_EPOCHS = 50       # Nombre max d'époques (avec early stopping)
LSTM_BATCH_SIZE = 32   # Taille des batchs pour l'entraînement

# --- Paramètres LightGBM (Gradient Boosting) ---
LGBM_PARAMS = {
    'objective': 'binary',          # Classification binaire (0 ou 1)
    'metric': 'binary_logloss',     # Fonction de perte
    'boosting_type': 'gbdt',        # Gradient Boosting classique
    'num_leaves': 31,               # Complexité des arbres (31 = optimal)
    'learning_rate': 0.05,          # Pas d'apprentissage
    'feature_fraction': 0.9,        # Utilise 90% des features par arbre
    'bagging_fraction': 0.8,        # Utilise 80% des données par arbre
    'bagging_freq': 5,              # Fréquence du bagging
    'verbose': -1,                  # Pas de logs (mode silencieux)
    'random_state': 42              # Pour reproductibilité
}

# Split train/test
TEST_SIZE = 0.2        # 20% des données pour le test (80% train)
RANDOM_STATE = 42      # Graine aléatoire pour reproductibilité


# ====================================
# 4. PARAMÈTRES DE L'APPLICATION STREAMLIT
# ====================================

# Seuils d'alerte pour classifier le niveau de risque
THRESHOLD_MODERATE = 0.15  # 15% : Risque modéré (affichage orange 🟠)
THRESHOLD_HIGH = 0.30      # 30% : Risque élevé (affichage rouge 🔴)
# < 15% = Risque faible (vert 🟢)
# 15-30% = Risque modéré (orange 🟠)
# > 30% = Risque élevé (rouge 🔴)

# Nombre d'heures d'historique à afficher dans les graphiques
HISTORICAL_HOURS = 24 * 7  # 1 semaine = 168 heures

# Valeurs par défaut des sliders dans l'interface Streamlit
# Ces valeurs correspondent à des conditions météo "normales" à Dakar
DEFAULT_TEMP = 25.0        # 25°C (température moyenne)
DEFAULT_HUMIDITE = 65.0    # 65% (humidité moyenne)
DEFAULT_VENT = 10.0        # 10 km/h (vent léger)
DEFAULT_CONSO = 800.0      # 800 MW (consommation moyenne)


# ====================================
# 5. BASE DE DONNÉES (FUTURE)
# ====================================
# Pour l'instant j'utilise SQLite (fichier local), mais le code est prêt
# pour basculer vers MySQL en production

# Type de base de données à utiliser
DATABASE_TYPE = 'mysql'  # 'sqlite' pour dev, 'mysql' pour prod

# SQLite : Base de données locale (fichier .db)
# Avantage : Simple, pas de serveur à installer
# Inconvénient : Mono-utilisateur, pas de concurrence
SQLITE_DB_FILE = DATA_DIR / "dakar_power.db"

# MySQL : Base de données distante (serveur)
# Avantage : Multi-utilisateurs, scalable
# Inconvénient : Nécessite un serveur MySQL
MYSQL_CONFIG = {
    'host': 'localhost',           # Serveur local (ou IP distante)
    'port': 3306,                  # Port par défaut MySQL
    'database': 'dakar_predictions',
    'user': 'root',
    'password': '',                # ← Vide en dev (à sécuriser en prod !)
    'charset': 'utf8mb4'           # Encodage UTF-8 complet
}

def get_db_connection_string():
    """
    Génère la chaîne de connexion SQLAlchemy selon le type de BDD.
    
    SQLAlchemy utilise des URLs au format :
    - SQLite : sqlite:///chemin/vers/fichier.db
    - MySQL : mysql+pymysql://user:password@host:port/database
    
    Returns:
        str: Chaîne de connexion SQLAlchemy
    
    Raises:
        ValueError: Si DATABASE_TYPE n'est ni 'sqlite' ni 'mysql'
    """
    if DATABASE_TYPE == 'sqlite':
        return f"sqlite:///{SQLITE_DB_FILE}"
    
    elif DATABASE_TYPE == 'mysql':
        # Construction de l'URL MySQL
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
# Dictionnaire pour mapper les noms de colonnes (pas utilisé actuellement,
# mais utile si je veux renommer les colonnes à l'avenir)

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
# Messages prédéfinis pour l'interface et les logs
# Utilisation d'emojis pour rendre les messages plus clairs

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
    """
    Affiche un résumé de la configuration du projet.
    
    Utile pour vérifier rapidement que tous les chemins sont corrects
    et que les paramètres sont bien configurés.
    
    Usage:
        python src/config.py
    """
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
    print("="*50)


# Si j'exécute ce fichier directement (python src/config.py),
# afficher la configuration
if __name__ == "__main__":
    print_config()