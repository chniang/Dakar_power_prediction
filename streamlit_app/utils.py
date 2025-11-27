# Fichier : streamlit_app/utils.py
# Fonctions utilitaires pour l'application Streamlit
# ====================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path
import sys

# Ajouter le répertoire parent au path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Imports du projet
from src.config import (
    LGBM_MODEL_FILE, LSTM_MODEL_FILE, SCALER_FILE, ENCODERS_FILE,
    QUARTIER_COORDS, THRESHOLD_MODERATE, THRESHOLD_HIGH,
    SEQUENCE_LENGTH, QUARTIERS_DAKAR
)

# Imports pour les visualisations
import plotly.graph_objects as go
import plotly.express as px

# ================================================================
# CHARGEMENT DES MODÈLES
# ================================================================

@st.cache_resource
def load_models_cached():
    """
    Charger les modèles ML avec mise en cache Streamlit
    
    Returns:
        dict: Dictionnaire contenant les modèles et leurs métadonnées
    """
    models = {}
    
    try:
        # Charger LightGBM
        if LGBM_MODEL_FILE.exists():
            lgbm_data = joblib.load(LGBM_MODEL_FILE)
            models['lgb'] = lgbm_data.get('model')
            models['lgb_threshold'] = lgbm_data.get('threshold', 0.5)
            st.success("✅ Modèle LightGBM chargé")
        else:
            st.error(f"❌ Fichier LightGBM introuvable : {LGBM_MODEL_FILE}")
            models['lgb'] = None
        
        # Charger LSTM
        if LSTM_MODEL_FILE.exists():
            from tensorflow import keras
            models['lstm'] = keras.models.load_model(str(LSTM_MODEL_FILE))
            
            # Charger le seuil LSTM
            threshold_file = LSTM_MODEL_FILE.parent / "lstm_threshold.txt"
            if threshold_file.exists():
                with open(threshold_file, 'r') as f:
                    models['lstm_threshold'] = float(f.read().strip())
            else:
                models['lstm_threshold'] = 0.5
            
            st.success("✅ Modèle LSTM chargé")
        else:
            st.error(f"❌ Fichier LSTM introuvable : {LSTM_MODEL_FILE}")
            models['lstm'] = None
        
        # Charger le scaler
        if SCALER_FILE.exists():
            models['scaler'] = joblib.load(SCALER_FILE)
        else:
            st.warning("⚠️ Scaler introuvable, création d'un scaler par défaut")
            from sklearn.preprocessing import StandardScaler
            models['scaler'] = StandardScaler()
        
        # Charger les encoders
        if ENCODERS_FILE.exists():
            models['encoders'] = joblib.load(ENCODERS_FILE)
        else:
            st.warning("⚠️ Encoders introuvables")
            models['encoders'] = {}
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des modèles : {e}")
        models = {'lgb': None, 'lstm': None, 'scaler': None, 'encoders': {}}
    
    return models

# ================================================================
# CRÉATION DES FEATURES TEMPORELLES
# ================================================================

def create_time_features(dt):
    """
    Créer les features temporelles à partir d'une date
    
    Args:
        dt (datetime): Date/heure
    
    Returns:
        dict: Dictionnaire des features temporelles
    """
    # Saison (hémisphère Nord)
    month = dt.month
    if month in [12, 1, 2]:
        saison = 0  # Hiver
    elif month in [3, 4, 5]:
        saison = 1  # Printemps
    elif month in [6, 7, 8]:
        saison = 2  # Été
    else:
        saison = 3  # Automne
    
    return {
        'heure': dt.hour,
        'jour_semaine': dt.weekday(),
        'mois': dt.month,
        'saison': saison,
        'est_weekend': 1 if dt.weekday() >= 5 else 0
    }

# ================================================================
# PRÉDICTION
# ================================================================

def make_prediction_single(models, quartier, temperature, humidite, vitesse_vent, consommation, temp_features):
    """
    Faire une prédiction unique
    
    Args:
        models (dict): Dictionnaire des modèles
        quartier (str): Nom du quartier
        temperature (float): Température en °C
        humidite (float): Humidité en %
        vitesse_vent (float): Vitesse du vent en km/h
        consommation (float): Consommation électrique en MW
        temp_features (dict): Features temporelles
    
    Returns:
        tuple: (prediction_lgb, prediction_lstm, risque_global) ou None
    """
    try:
        # Créer le vecteur de features
        features = np.array([[
            temperature,
            humidite,
            vitesse_vent,
            consommation,
            temp_features['heure'],
            temp_features['jour_semaine'],
            temp_features['mois'],
            temp_features['saison'],
            temp_features['est_weekend']
        ]])
        
        # Prédiction LightGBM
        pred_lgb = 0.0
        if models.get('lgb') is not None:
            try:
                pred_lgb_raw = models['lgb'].predict(features)[0]
                pred_lgb = float(pred_lgb_raw * 100)
            except Exception as e:
                st.warning(f"⚠️ Erreur LightGBM : {e}")
                pred_lgb = 0.0
        
        # Prédiction LSTM
        pred_lstm = 0.0
        if models.get('lstm') is not None:
            try:
                # Créer une séquence (répéter les features pour simuler une séquence)
                sequence = np.repeat(features, SEQUENCE_LENGTH, axis=0)
                sequence = sequence.reshape(1, SEQUENCE_LENGTH, features.shape[1])
                
                pred_lstm_raw = models['lstm'].predict(sequence, verbose=0)[0][0]
                pred_lstm = float(pred_lstm_raw * 100)
            except Exception as e:
                st.warning(f"⚠️ Erreur LSTM : {e}")
                pred_lstm = 0.0
        
        # Calculer le risque global (moyenne pondérée)
        if pred_lgb > 0 and pred_lstm > 0:
            risque_global = (pred_lgb * 0.6 + pred_lstm * 0.4)
        elif pred_lgb > 0:
            risque_global = pred_lgb
        elif pred_lstm > 0:
            risque_global = pred_lstm
        else:
            risque_global = 0.0
        
        return pred_lgb, pred_lstm, risque_global
    
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
        return None

# ================================================================
# VISUALISATIONS
# ================================================================

def create_gauge_chart(value, title):
    """
    Créer une jauge de risque
    
    Args:
        value (float): Valeur du risque (0-100)
        title (str): Titre de la jauge
    
    Returns:
        plotly.graph_objects.Figure: Figure Plotly
    """
    # Déterminer la couleur selon le risque
    if value < THRESHOLD_MODERATE:
        color = "green"
    elif value < THRESHOLD_HIGH:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': f"Risque de Coupure - {title}", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, THRESHOLD_MODERATE], 'color': 'lightgreen'},
                {'range': [THRESHOLD_MODERATE, THRESHOLD_HIGH], 'color': 'lightyellow'},
                {'range': [THRESHOLD_HIGH, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': THRESHOLD_HIGH
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_risk_map(df_predictions):
    """
    Créer une carte interactive des risques
    
    Args:
        df_predictions (pd.DataFrame): DataFrame avec les prédictions par quartier
    
    Returns:
        plotly.graph_objects.Figure: Figure Plotly
    """
    # Ajouter les coordonnées
    df_map = df_predictions.copy()
    df_map['lat'] = df_map['Quartier'].map(lambda q: QUARTIER_COORDS.get(q, {}).get('lat', 14.7))
    df_map['lon'] = df_map['Quartier'].map(lambda q: QUARTIER_COORDS.get(q, {}).get('lon', -17.45))
    
    # Créer la carte
    fig = px.scatter_mapbox(
        df_map,
        lat='lat',
        lon='lon',
        size='Risque Global',
        color='Risque Global',
        hover_name='Quartier',
        hover_data={
            'LightGBM': ':.1f',
            'LSTM': ':.1f',
            'Risque Global': ':.1f',
            'lat': False,
            'lon': False
        },
        color_continuous_scale='RdYlGn_r',
        size_max=50,
        zoom=10,
        mapbox_style='open-street-map',
        title="Carte des Risques de Coupure par Quartier"
    )
    
    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    
    return fig

def get_risk_color(risk_value):
    """
    Obtenir la couleur associée à un niveau de risque
    
    Args:
        risk_value (float): Valeur du risque (0-100)
    
    Returns:
        str: Code couleur
    """
    if risk_value < THRESHOLD_MODERATE:
        return "#28a745"  # Vert
    elif risk_value < THRESHOLD_HIGH:
        return "#ffc107"  # Orange
    else:
        return "#dc3545"  # Rouge

def get_risk_level(risk_value):
    """
    Obtenir le niveau de risque textuel
    
    Args:
        risk_value (float): Valeur du risque (0-100)
    
    Returns:
        str: Niveau de risque
    """
    if risk_value < THRESHOLD_MODERATE:
        return "FAIBLE"
    elif risk_value < THRESHOLD_HIGH:
        return "MOYEN"
    else:
        return "ÉLEVÉ"