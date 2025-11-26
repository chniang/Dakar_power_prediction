# Fichier : streamlit_app/app.py
# Application Streamlit Professionnelle - Dakar Power Prediction
# ================================================================

# Importations des librairies standards et scientifiques
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour permettre les imports depuis 'src' et 'streamlit_app'
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Imports locaux (fonctions utilitaires pour le chargement, la prédiction et les données)
from streamlit_app.utils import (
    load_models, get_database, make_prediction_single,
    get_historical_data, get_statistics_by_quartier,
    validate_input, get_quartier_coords, get_quartier_list,
    save_prediction_to_db, format_percentage
)

# Assurez-vous que src.config est accessible et charger les seuils de risque et le centre de la carte
try:
    from src.config import THRESHOLD_MODERATE, THRESHOLD_HIGH, MAP_CENTER
except ImportError:
    # Valeurs par défaut si le fichier config n'est pas trouvé (pour la robustesse de l'app)
    THRESHOLD_MODERATE = 0.15
    THRESHOLD_HIGH = 0.30
    MAP_CENTER = {"lat": 14.716677, "lon": -17.467686}
    st.warning("⚠️ Impossible de charger src.config. Utilisation des seuils par défaut.")

# ====================================
# CONFIGURATION DE LA PAGE
# ====================================

st.set_page_config(
    page_title="Dakar Power Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        font-size: 1.1rem;
    }
    .debug-box {
        background-color: #f0f8ff;
        color: #1c1e21;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .stApp[data-theme="dark"] .debug-box {
        background-color: #262730;
        color: #f0f8ff;
        border-left: 4px solid #90caf9;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ====================================
# CHARGEMENT DES RESSOURCES (CACHE)
# ====================================

@st.cache_resource
def load_models_cached():
    """✅ CORRECTION : Gestion d'erreur améliorée pour le chargement LSTM"""
    try:
        return load_models()
    except Exception as e:
        st.error(f"❌ Erreur critique lors du chargement des modèles : {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, 0.5, None, 0.5, None, None

@st.cache_data
def load_static_data():
    """Charge les données statiques (quartiers et coordonnées)."""
    quartiers = get_quartier_list()
    coords = get_quartier_coords()
    return {'quartiers': quartiers, 'coords': coords}

# ====================================
# FONCTION DE PRÉDICTION POUR TOUS LES QUARTIERS
# ====================================

def get_predictions_for_all_quartiers(temp, hum, vent, conso, models_data, quartiers_list, run_id):
    """Calcule les prédictions de risque pour tous les quartiers."""
    print(f"🔍 [RUN {run_id}] Calcul predictions: temp={temp}°C, hum={hum}%, vent={vent}km/h, conso={conso}MW")
    
    quartiers_data = []
    
    for quartier in quartiers_list:
        coords = models_data['coords'].get(quartier, {'lat': 0, 'lon': 0})
        
        input_data = {
            'temperature': temp,
            'humidite': hum,
            'vent': vent,
            'consommation': conso
        }
        
        historical_data = get_historical_data(models_data['db'], quartier=quartier, hours=168)
        
        try:
            result = make_prediction_single(
                input_data,
                quartier,
                models_data['lgbm_model'],
                models_data['lgbm_threshold'],
                models_data['lstm_model'],
                models_data['lstm_threshold'],
                models_data['scaler'],
                models_data['label_encoder'],
                historical_data
            )
            
            print(f"  {quartier}: LightGBM={result['proba_lgbm']*100:.2f}%, LSTM={result['proba_lstm']*100:.2f}%")
            
            quartiers_data.append({
                'Quartier': quartier,
                'Latitude': coords['lat'],
                'Longitude': coords['lon'],
                'Probabilité': result['proba_moyenne'],
                'Statut': result['statut'],
                'Emoji': result['emoji'],
                'LightGBM': result['proba_lgbm'],
                'LSTM': result['proba_lstm']
            })
        except Exception as e:
            print(f"  ❌ Erreur pour {quartier}: {e}")
            quartiers_data.append({
                'Quartier': quartier,
                'Latitude': coords['lat'],
                'Longitude': coords['lon'],
                'Probabilité': 0.0,
                'Statut': 'Erreur',
                'Emoji': '⚠️',
                'LightGBM': 0.0,
                'LSTM': 0.0
            })
    
    print(f"✅ [RUN {run_id}] Calcul terminé pour {len(quartiers_data)} quartiers")
    return quartiers_data

# ====================================
# CHARGEMENT INITIAL & SESSION STATE
# ====================================

models_result = load_models_cached()

if models_result is None or models_result[0] is None:
    st.error("❌ Échec du chargement des modèles. L'application ne peut pas fonctionner correctement.")
    st.stop()

lgbm_model, lgbm_threshold, lstm_model, lstm_threshold, scaler, label_encoder = models_result
static_data = load_static_data()
db = get_database()

data = {
    'lgbm_model': lgbm_model,
    'lgbm_threshold': lgbm_threshold,
    'lstm_model': lstm_model,
    'lstm_threshold': lstm_threshold,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'db': db,
    'quartiers': static_data['quartiers'],
    'coords': static_data['coords']
}

if 'last_prediction_result' not in st.session_state:
    st.session_state['last_prediction_result'] = None
if 'last_prediction_quartier' not in st.session_state:
    st.session_state['last_prediction_quartier'] = None

# ====================================
# HEADER
# ====================================

st.markdown('<div class="main-header">⚡ Prédiction de Coupures d\'Électricité à Dakar</div>', unsafe_allow_html=True)
st.markdown("---")

# ====================================
# SIDEBAR - CONTRÔLES
# ====================================

with st.sidebar:
    st.title("⚙️ Paramètres")
    st.markdown("---")
    
    st.subheader("📍 Quartier à analyser")
    selected_quartier = st.selectbox(
        "Choisissez un quartier",
        options=data['quartiers'],
        index=0,
        key="selected_quartier_sidebar",
        help="Sélectionnez le quartier pour la prédiction immédiate"
    )
    
    st.markdown("---")
    st.subheader("🌡️ Conditions Météorologiques")
    
    temperature = st.slider("Température (°C)", 15.0, 40.0, 25.0, 0.5, key="temp_slider")
    humidite = st.slider("Humidité (%)", 30.0, 100.0, 65.0, 1.0, key="hum_slider")
    vent = st.slider("Vitesse du vent (km/h)", 0.0, 50.0, 15.0, 0.5, key="vent_slider")
    
    st.markdown("---")
    st.subheader("⚡ Consommation Électrique")
    consommation = st.slider("Consommation (MW)", 200.0, 1500.0, 800.0, 10.0, key="conso_slider")
    
    st.markdown("---")
    predict_button = st.button("🔮 Lancer la Prédiction", type="primary", use_container_width=True)
    
    st.markdown("---")
    with st.expander("ℹ️ À propos"):
        st.markdown(f"""
        **Dakar Power Prediction**
        
        Application de prédiction des coupures d'électricité à Dakar utilisant :
        - 🌳 LightGBM (Machine Learning)
        - 🧠 LSTM (Deep Learning)
        - 🗄️ Base de données MySQL
        
        **Seuils de risque :**
        - 🟢 Faible : < {THRESHOLD_MODERATE*100:.0f}%
        - 🟠 Modéré : {THRESHOLD_MODERATE*100:.0f}% - {THRESHOLD_HIGH*100:.0f}%
        - 🔴 Élevé : > {THRESHOLD_HIGH*100:.0f}%
        """)

# ====================================
# TABS PRINCIPAUX
# ====================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Prédiction Immédiate",
    "🗺️ Carte Interactive",
    "📊 Analyse par Quartier",
    "📈 Historique & Tendances"
])

# ====================================
# FONCTION D'AFFICHAGE
# ====================================

def display_single_prediction(result, quartier):
    """Affiche les résultats détaillés de la prédiction."""
    st.success("✅ Prédiction effectuée avec succès !")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.metric(
            label=f"🎯 Probabilité de Coupure - {quartier}",
            value=format_percentage(result['proba_moyenne']),
            delta=f"{result['statut']} {result['emoji']}",
            delta_color="off"
        )
    
    with col2:
        st.metric(
            label="🌳 LightGBM",
            value=format_percentage(result['proba_lgbm']),
            help=f"Seuil de décision : {result['seuil_lgbm']:.3f}"
        )
    
    with col3:
        if result['lstm_utilisable']:
            value_lstm = format_percentage(result['proba_lstm'])
            help_text = f"Seuil de décision : {result['seuil_lstm']:.3f}"
        else:
            value_lstm = "N/A"
            help_text = "Historique insuffisant pour LSTM (moins de 24h)"
        
        st.metric(label="🧠 LSTM", value=value_lstm, help=help_text)
    
    st.markdown("---")
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result['proba_moyenne'] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Niveau de Risque (%)", 'font': {'size': 24}},
        delta={'reference': THRESHOLD_MODERATE * 100},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': result['color']},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, THRESHOLD_MODERATE * 100], 'color': 'lightgreen'},
                {'range': [THRESHOLD_MODERATE * 100, THRESHOLD_HIGH * 100], 'color': 'lightyellow'},
                {'range': [THRESHOLD_HIGH * 100, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': result['proba_moyenne'] * 100
            }
        }
    ))
    
    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, width='stretch')

# ====================================
# TAB 1 : PRÉDICTION IMMÉDIATE
# ====================================

with tab1:
    st.header("🎯 Prédiction Immédiate")
    
    if predict_button:
        is_valid, error_msg = validate_input(temperature, humidite, vent, consommation)
        
        if not is_valid:
            st.error(error_msg)
            st.session_state['last_prediction_result'] = None
        else:
            with st.spinner(f"🔄 Calcul de la prédiction pour {selected_quartier} en cours..."):
                input_data = {
                    'temperature': temperature,
                    'humidite': humidite,
                    'vent': vent,
                    'consommation': consommation
                }
                
                historical_data = get_historical_data(data['db'], quartier=selected_quartier, hours=24)
                
                try:
                    result = make_prediction_single(
                        input_data,
                        selected_quartier,
                        data['lgbm_model'],
                        data['lgbm_threshold'],
                        data['lstm_model'],
                        data['lstm_threshold'],
                        data['scaler'],
                        data['label_encoder'],
                        historical_data
                    )
                    
                    st.session_state['last_prediction_result'] = result
                    st.session_state['last_prediction_quartier'] = selected_quartier
                    
                    display_single_prediction(result, selected_quartier)
                    
                    if data['db'] is not None:
                        try:
                            prediction_data = {
                                'date_heure': datetime.now(),
                                'quartier': selected_quartier,
                                'temp_celsius': temperature,
                                'humidite_percent': humidite,
                                'vitesse_vent': vent,
                                'conso_megawatt': consommation,
                                'proba_lgbm': result['proba_lgbm'],
                                'proba_lstm': result['proba_lstm'],
                                'proba_moyenne': result['proba_moyenne'],
                                'prediction': 1 if result['proba_moyenne'] >= THRESHOLD_MODERATE else 0,
                                'modele_utilise': 'ensemble' if result['lstm_utilisable'] else 'lgbm_only',
                                'seuil_decision': THRESHOLD_MODERATE
                            }
                            pred_id = save_prediction_to_db(data['db'], prediction_data)
                            if pred_id:
                                st.info(f"💾 Prédiction #{pred_id} sauvegardée dans la base de données")
                        except Exception as e:
                            st.warning(f"⚠️ Impossible de sauvegarder la prédiction: {e}")
                            
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction : {e}")
                    import traceback
                    st.error(traceback.format_exc())
    
    elif st.session_state['last_prediction_result'] is not None:
        display_single_prediction(st.session_state['last_prediction_result'], st.session_state['last_prediction_quartier'])
    else:
        st.info("👈 Configurez les paramètres dans la barre latérale et cliquez sur **'Lancer la Prédiction'**")
        
        st.markdown("### 📋 Exemple de conditions")
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            st.markdown("""
            **Conditions normales :**
            * Température : 25°C
            * Humidité : 65%
            * Vent : 15 km/h
            * Consommation : 800 MW
            """)
        
        with example_col2:
            st.markdown("""
            **Conditions à risque :**
            * Température : 38°C (canicule)
            * Humidité : 85%
            * Vent : 45 km/h (tempête)
            * Consommation : 1200 MW (surcharge)
            """)

# ====================================
# TAB 2 : CARTE INTERACTIVE
# ====================================

with tab2:
    st.header("🗺️ Carte Interactive des Risques")
    
    col_header1, col_header2 = st.columns([4, 1])
    with col_header2:
        refresh_button = st.button("🔄 Rafraîchir", key="refresh_map", type="secondary", use_container_width=True)
    
    run_id = f"{temperature}_{humidite}_{vent}_{consommation}"
    
    st.markdown(f"""
    <div class="debug-box">
        <strong>🔍 Conditions actuelles utilisées pour les prédictions :</strong><br>
        🌡️ Température : <strong>{temperature}°C</strong> | 
        💧 Humidité : <strong>{humidite}%</strong> | 
        💨 Vent : <strong>{vent} km/h</strong> | 
        ⚡ Consommation : <strong>{consommation} MW</strong>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🔄 Calcul des prédictions en temps réel pour tous les quartiers..."):
        quartiers_data = get_predictions_for_all_quartiers(
            temperature, humidite, vent, consommation,
            data, data['quartiers'], run_id
        )
        
        df_map = pd.DataFrame(quartiers_data)
        
        fig_map = px.scatter_mapbox(
            df_map,
            lat='Latitude',
            lon='Longitude',
            size='Probabilité',
            color='Probabilité',
            hover_name='Quartier',
            hover_data={
                'Probabilité': ':.2%',
                'Statut': True,
                'LightGBM': ':.2%',
                'LSTM': ':.2%',
                'Latitude': False,
                'Longitude': False
            },
            color_continuous_scale='RdYlGn_r',
            size_max=30,
            zoom=11,
            center={'lat': MAP_CENTER['lat'], 'lon': MAP_CENTER['lon']},
            mapbox_style='open-street-map',
            title=f"Risque de Coupure par Quartier (Temp: {temperature}°C, Conso: {consommation}MW)"
        )
        
        fig_map.update_layout(
            height=600,
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title="Probabilité", tickformat='.0%')
        )
        
        st.plotly_chart(fig_map, width='stretch')
        
        st.markdown("### 📊 Récapitulatif par Quartier")
        
        df_display = df_map.copy()
        df_display = df_display.sort_values('Probabilité', ascending=False)
        
        df_display['Probabilité'] = df_display['Probabilité'].apply(lambda x: f"{x*100:.2f}%")
        df_display['LightGBM'] = df_display['LightGBM'].apply(lambda x: f"{x*100:.2f}%")
        df_display['LSTM'] = df_display['LSTM'].apply(lambda x: f"{x*100:.2f}%" if x > 0.001 else "N/A")
        
        df_display = df_display[['Quartier', 'Probabilité', 'Statut', 'LightGBM', 'LSTM']]
        
        st.dataframe(df_display, width='stretch', hide_index=True)
        
        st.info(f"💡 **Prédictions calculées en temps réel** | Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")

# ====================================
# TAB 3 : ANALYSE PAR QUARTIER
# ====================================

with tab3:
    st.header("📊 Analyse par Quartier")
    
    if data['db'] is not None:
        with st.spinner("📊 Chargement des statistiques historiques..."):
            stats_df = get_statistics_by_quartier(data['db'])
            
            if not stats_df.empty:
                fig_bar = px.bar(
                    stats_df.sort_values('taux_coupure', ascending=False),
                    x='quartier',
                    y='taux_coupure',
                    title="Taux de Coupure Historique par Quartier (Basé sur les données enregistrées)",
                    labels={'taux_coupure': 'Taux de Coupure (%)', 'quartier': 'Quartier'},
                    color='taux_coupure',
                    color_continuous_scale='Reds',
                    text='taux_coupure'
                )
                
                fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig_bar.update_layout(height=400, showlegend=False, xaxis={'categoryorder': 'total descending'})
                
                st.plotly_chart(fig_bar, width='stretch')
                
                st.markdown("### 📋 Statistiques Détaillées")
                
                stats_display = stats_df.copy()
                stats_display['taux_coupure'] = stats_display['taux_coupure'].apply(lambda x: f"{x:.2f}%")
                stats_display['temp_moyenne'] = stats_display['temp_moyenne'].apply(lambda x: f"{x:.1f}°C")
                stats_display['conso_moyenne'] = stats_display['conso_moyenne'].apply(lambda x: f"{x:.1f} MW")
                stats_display.columns = ['Quartier', 'Total Enregistrements', 'Total Coupures', 'Taux Coupure', 'Temp. Moyenne', 'Conso. Moyenne']
                
                st.dataframe(stats_display, width='stretch', hide_index=True)
            else:
                st.warning("⚠️ Aucune donnée statistique disponible dans la base de données.")
    else:
        st.error("❌ Base de données non disponible. Impossible de charger les statistiques.")

# ====================================
# TAB 4 : HISTORIQUE & TENDANCES
# ====================================

with tab4:
    st.header("📈 Historique & Tendances")
    
    if data['db'] is not None:
        quartier_histo = st.selectbox(
            "Sélectionnez un quartier pour l'historique",
            options=data['quartiers'],
            key='quartier_histo'
        )
        
        col_period1, col_period2 = st.columns(2)
        with col_period1:
            hours_back = st.slider("Heures d'historique à afficher (Max 1 semaine)", 24, 168, 168, 24)
        
        with st.spinner(f"📊 Chargement de {hours_back}h d'historique pour {quartier_histo}..."):
            hist_df = get_historical_data(data['db'], quartier=quartier_histo, hours=hours_back)
            
            if not hist_df.empty:
                fig_hist = go.Figure()
                
                fig_hist.add_trace(go.Scatter(
                    x=hist_df['date_heure'],
                    y=hist_df['conso_megawatt'],
                    name='Consommation (MW)',
                    line=dict(color='blue', width=2),
                    yaxis='y1'
                ))
                
                fig_hist.add_trace(go.Scatter(
                    x=hist_df['date_heure'],
                    y=hist_df['temp_celsius'],
                    name='Température (°C)',
                    yaxis='y2',
                    line=dict(color='orange', width=2, dash='dot')
                ))
                
                coupures_df = hist_df[hist_df['coupure'] == 1]
                if not coupures_df.empty:
                    fig_hist.add_trace(go.Scatter(
                        x=coupures_df['date_heure'],
                        y=coupures_df['conso_megawatt'],
                        mode='markers',
                        name='Coupure Réelle',
                        marker=dict(color='red', size=10, symbol='x'),
                        yaxis='y1'
                    ))
                
                fig_hist.update_layout(
                    title=f"Historique Récent - {quartier_histo}",
                    xaxis_title="Date et Heure",
                    yaxis=dict(
                        title="Consommation (MW)",
                        titlefont=dict(color='blue'),
                        tickfont=dict(color='blue')
                    ),
                    yaxis2=dict(
                        title="Température (°C)",
                        titlefont=dict(color='orange'),
                        tickfont=dict(color='orange'),
                        overlaying='y',
                        side='right'
                    ),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_hist, width='stretch')
                
                st.markdown("### 🔍 Statistiques de la Période Sélectionnée")
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    st.metric("📊 Enregistrements", len(hist_df))
                
                with col_stat2:
                    nb_coupures = hist_df['coupure'].sum()
                    st.metric("⚡ Coupures (enregistrées)", int(nb_coupures))
                
                with col_stat3:
                    taux = hist_df['coupure'].mean() * 100
                    st.metric("📈 Taux de Coupure", f"{taux:.2f}%")
                
                with col_stat4:
                    temp_moy = hist_df['temp_celsius'].mean()
                    st.metric("🌡️ Temp. Moy. (période)", f"{temp_moy:.1f}°C")
            else:
                st.warning(f"⚠️ Aucune donnée historique disponible pour {quartier_histo} sur les dernières {hours_back} heures.")
    else:
        st.error("❌ Base de données non disponible. Impossible de charger l'historique.")

# ====================================
# FOOTER
# ====================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>⚡ <strong>Dakar Power Prediction</strong> | Développé avec Streamlit, LightGBM, LSTM & MySQL</p>
    <p>📊 Données: 52,704 enregistrements | 🏘️ Quartiers: 6</p>
</div>
""", unsafe_allow_html=True)