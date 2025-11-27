# Fichier : streamlit_app/app.py
# Application Streamlit Professionnelle - Dakar Power Prediction avec Supabase
# ================================================================

# Importations des librairies standards et scientifiques
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour importer les modules
sys.path.append(str(Path(__file__).parent.parent))

# Imports du projet
from src.config import QUARTIERS_DAKAR, SEUILS_RISQUE
from streamlit_app.utils import (
    load_models_cached,
    create_time_features,
    make_prediction_single,
    create_gauge_chart,
    create_risk_map,
    get_risk_color,
    get_risk_level
)

# Import Supabase Database
from src.database import get_db

# Imports pour les visualisations
import plotly.express as px
import plotly.graph_objects as go

# ================================================================
# CONFIGURATION DE LA PAGE
# ================================================================

st.set_page_config(
    page_title="⚡ Dakar Power Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8C00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .risk-high {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8C00 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# HEADER
# ================================================================

st.markdown('<h1 class="main-header">⚡ Prédiction de Coupures d\'Électricité à Dakar</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Système de prédiction intelligent basé sur LightGBM et LSTM</p>', unsafe_allow_html=True)

# ================================================================
# CHARGEMENT DES MODÈLES ET BASE DE DONNÉES
# ================================================================

@st.cache_resource
def initialize_app():
    """Initialiser les modèles et la base de données"""
    models = load_models_cached()
    db = get_db()
    return models, db

models, db = initialize_app()

# Vérifier que les modèles sont chargés
if not models or models['lgb'] is None:
    st.error("❌ Erreur : Les modèles n'ont pas pu être chargés. Vérifiez le dossier 'models/'.")
    st.stop()

# ================================================================
# SIDEBAR - PARAMÈTRES
# ================================================================

st.sidebar.header("⚙️ Paramètres")

# Sélection du quartier
st.sidebar.subheader("📍 Quartier à analyser")
quartier = st.sidebar.selectbox(
    "Choisissez un quartier",
    QUARTIERS_DAKAR,
    index=0
)

# Conditions météorologiques
st.sidebar.subheader("🌡️ Conditions Météorologiques")

temperature = st.sidebar.slider(
    "Température (°C)",
    min_value=15.0,
    max_value=40.0,
    value=25.0,
    step=0.5,
    help="Température ambiante en degrés Celsius"
)

humidite = st.sidebar.slider(
    "Humidité (%)",
    min_value=30.0,
    max_value=100.0,
    value=65.0,
    step=1.0,
    help="Taux d'humidité relative"
)

vitesse_vent = st.sidebar.slider(
    "Vitesse du vent (km/h)",
    min_value=0.0,
    max_value=50.0,
    value=15.0,
    step=1.0,
    help="Vitesse moyenne du vent"
)

# Consommation électrique
st.sidebar.subheader("⚡ Consommation Électrique")

consommation = st.sidebar.slider(
    "Consommation (MW)",
    min_value=200.0,
    max_value=1500.0,
    value=800.0,
    step=10.0,
    help="Consommation électrique estimée"
)

# Bouton de prédiction
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔮 Lancer la Prédiction", type="primary")

# Informations
st.sidebar.markdown("---")
st.sidebar.info("""
ℹ️ **À propos**

Ce système utilise :
- 🤖 **LightGBM** : Algorithme de boosting
- 🧠 **LSTM** : Réseau de neurones récurrent
- 🗺️ **6 quartiers** de Dakar analysés
- 📊 **Données temps réel** sauvegardées dans Supabase
""")

# ================================================================
# TABS PRINCIPAUX
# ================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Prédiction Immédiate",
    "🗺️ Carte Interactive",
    "📊 Analyse par Quartier",
    "📈 Historique & Tendances"
])

# ================================================================
# TAB 1 : PRÉDICTION IMMÉDIATE
# ================================================================

with tab1:
    st.header("🎯 Prédiction Immédiate")
    
    if predict_button:
        with st.spinner("🔄 Calcul de la prédiction en cours..."):
            # Créer les features temporelles
            now = datetime.now()
            temp_features = create_time_features(now)
            
            # Faire la prédiction
            result = make_prediction_single(
                models=models,
                quartier=quartier,
                temperature=temperature,
                humidite=humidite,
                vitesse_vent=vitesse_vent,
                consommation=consommation,
                temp_features=temp_features
            )
            
            if result:
                pred_lgb, pred_lstm, risque_global = result
                
                # Sauvegarder dans Supabase
                try:
                    db.save_prediction(
                        quartier=quartier,
                        temp=temperature,
                        hum=humidite,
                        vent=vitesse_vent,
                        conso=consommation,
                        pred_lgb=pred_lgb,
                        pred_lstm=pred_lstm,
                        risque_global=risque_global
                    )
                except Exception as e:
                    st.warning(f"⚠️ Prédiction réussie mais non sauvegardée : {e}")
                
                # Afficher la jauge de risque
                st.subheader(f"📍 Résultat pour {quartier}")
                
                fig_gauge = create_gauge_chart(risque_global, quartier)
                st.plotly_chart(fig_gauge, width='stretch')
                
                # Métriques détaillées
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="🤖 Prédiction LightGBM",
                        value=f"{pred_lgb:.1f}%",
                        delta=f"{pred_lgb - 50:.1f}% vs moyenne"
                    )
                
                with col2:
                    st.metric(
                        label="🧠 Prédiction LSTM",
                        value=f"{pred_lstm:.1f}%",
                        delta=f"{pred_lstm - 50:.1f}% vs moyenne"
                    )
                
                with col3:
                    risk_level = get_risk_level(risque_global)
                    st.metric(
                        label="⚠️ Niveau de Risque",
                        value=risk_level,
                        delta=f"{risque_global:.1f}%"
                    )
                
                # Interprétation
                st.markdown("---")
                st.subheader("💡 Interprétation")
                
                if risque_global < SEUILS_RISQUE['faible']:
                    st.success(f"""
                    ✅ **Risque FAIBLE** ({risque_global:.1f}%)
                    
                    Les conditions actuelles sont favorables. Risque de coupure très faible.
                    """)
                elif risque_global < SEUILS_RISQUE['moyen']:
                    st.warning(f"""
                    ⚠️ **Risque MOYEN** ({risque_global:.1f}%)
                    
                    Conditions à surveiller. Une vigilance est recommandée.
                    """)
                else:
                    st.error(f"""
                    🚨 **Risque ÉLEVÉ** ({risque_global:.1f}%)
                    
                    Conditions critiques ! Risque important de coupure d'électricité.
                    """)
                
                # Facteurs contributifs
                st.markdown("---")
                st.subheader("📊 Conditions Actuelles")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **🌡️ Météo :**
                    - Température : {temperature}°C
                    - Humidité : {humidite}%
                    - Vent : {vitesse_vent} km/h
                    """)
                
                with col2:
                    st.markdown(f"""
                    **⚡ Électricité :**
                    - Consommation : {consommation} MW
                    - Quartier : {quartier}
                    - Heure : {now.strftime('%H:%M')}
                    """)
            
            else:
                st.error("❌ Erreur lors de la prédiction. Vérifiez les paramètres.")
    
    else:
        st.info("👈 Configurez les paramètres dans la barre latérale et cliquez sur 'Lancer la Prédiction'")
        
        # Exemples de conditions
        st.markdown("---")
        st.subheader("📋 Exemple de conditions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Conditions normales :**
            - 🌡️ Température : 25°C
            - 💧 Humidité : 65%
            - 🌬️ Vent : 15 km/h
            - ⚡ Consommation : 800 MW
            """)
        
        with col2:
            st.markdown("""
            **Conditions à risque :**
            - 🌡️ Température : 38°C (canicule)
            - 💧 Humidité : 85%
            - 🌬️ Vent : 45 km/h (tempête)
            - ⚡ Consommation : 1200 MW (surcharge)
            """)

# ================================================================
# TAB 2 : CARTE INTERACTIVE
# ================================================================

with tab2:
    st.header("🗺️ Carte Interactive")
    
    if predict_button or st.button("🔄 Calculer pour tous les quartiers", key="map_calc"):
        with st.spinner("🔄 Calcul des prédictions pour tous les quartiers..."):
            now = datetime.now()
            temp_features = create_time_features(now)
            
            predictions_data = []
            
            for q in QUARTIERS_DAKAR:
                result = make_prediction_single(
                    models=models,
                    quartier=q,
                    temperature=temperature,
                    humidite=humidite,
                    vitesse_vent=vitesse_vent,
                    consommation=consommation,
                    temp_features=temp_features
                )
                
                if result:
                    pred_lgb, pred_lstm, risque_global = result
                    predictions_data.append({
                        'Quartier': q,
                        'LightGBM': pred_lgb,
                        'LSTM': pred_lstm,
                        'Risque Global': risque_global,
                        'Niveau': get_risk_level(risque_global)
                    })
                    
                    # Sauvegarder dans Supabase
                    try:
                        db.save_prediction(
                            quartier=q,
                            temp=temperature,
                            hum=humidite,
                            vent=vitesse_vent,
                            conso=consommation,
                            pred_lgb=pred_lgb,
                            pred_lstm=pred_lstm,
                            risque_global=risque_global
                        )
                    except:
                        pass  # Silencieux pour ne pas bloquer l'affichage
            
            if predictions_data:
                df_predictions = pd.DataFrame(predictions_data)
                
                # Carte interactive
                fig_map = create_risk_map(df_predictions)
                st.plotly_chart(fig_map, width='stretch')
                
                # Tableau des résultats
                st.markdown("---")
                st.subheader("📊 Résultats par Quartier")
                
                # Formater le dataframe
                df_display = df_predictions.copy()
                df_display['LightGBM'] = df_display['LightGBM'].apply(lambda x: f"{x:.1f}%")
                df_display['LSTM'] = df_display['LSTM'].apply(lambda x: f"{x:.1f}%")
                df_display['Risque Global'] = df_display['Risque Global'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(df_display, width='stretch', hide_index=True)
                
                # Statistiques globales
                st.markdown("---")
                st.subheader("📈 Statistiques Globales")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    risque_moyen = df_predictions['Risque Global'].mean()
                    st.metric("Risque Moyen", f"{risque_moyen:.1f}%")
                
                with col2:
                    risque_max = df_predictions['Risque Global'].max()
                    quartier_max = df_predictions.loc[df_predictions['Risque Global'].idxmax(), 'Quartier']
                    st.metric("Risque Maximum", f"{risque_max:.1f}%", delta=quartier_max)
                
                with col3:
                    nb_critique = len(df_predictions[df_predictions['Risque Global'] >= SEUILS_RISQUE['eleve']])
                    st.metric("Quartiers Critiques", nb_critique)
                
                with col4:
                    nb_securise = len(df_predictions[df_predictions['Risque Global'] < SEUILS_RISQUE['faible']])
                    st.metric("Quartiers Sécurisés", nb_securise)
    
    else:
        st.info("👈 Cliquez sur 'Calculer pour tous les quartiers' ou lancez une prédiction dans la sidebar")

# ================================================================
# TAB 3 : ANALYSE PAR QUARTIER
# ================================================================

with tab3:
    st.header("📊 Analyse par Quartier")
    
    # Récupérer les statistiques depuis Supabase
    stats_quartiers = db.get_quartier_stats()
    
    if not stats_quartiers.empty:
        st.subheader("📈 Statistiques Cumulées")
        
        # Formater le dataframe
        stats_display = stats_quartiers.copy()
        stats_display['taux_risque'] = stats_display['taux_risque'].apply(lambda x: f"{x:.1f}%")
        stats_display['derniere_maj'] = pd.to_datetime(stats_display['derniere_maj']).dt.strftime('%Y-%m-%d %H:%M')
        
        stats_display.columns = ['Quartier', 'Total Prédictions', 'Coupures Prédites', 'Taux de Risque', 'Dernière MAJ']
        
        st.dataframe(stats_display, width='stretch', hide_index=True)
        
        # Graphiques
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique en barres - Total prédictions
            fig_bar = px.bar(
                stats_quartiers,
                x='quartier',
                y='total_predictions',
                title="Nombre de Prédictions par Quartier",
                labels={'quartier': 'Quartier', 'total_predictions': 'Nombre de Prédictions'},
                color='total_predictions',
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, width='stretch')
        
        with col2:
            # Graphique en barres - Taux de risque
            fig_risk = px.bar(
                stats_quartiers,
                x='quartier',
                y='taux_risque',
                title="Taux de Risque par Quartier",
                labels={'quartier': 'Quartier', 'taux_risque': 'Taux de Risque (%)'},
                color='taux_risque',
                color_continuous_scale='Reds'
            )
            fig_risk.update_layout(showlegend=False)
            st.plotly_chart(fig_risk, width='stretch')
        
    else:
        st.info("📊 Aucune statistique disponible. Lancez quelques prédictions pour générer des données !")

# ================================================================
# TAB 4 : HISTORIQUE & TENDANCES
# ================================================================

with tab4:
    st.header("📈 Historique & Tendances")
    
    # Sélection de la période
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        start_date = st.date_input(
            "Date de début",
            value=datetime.now() - timedelta(days=7),
            max_value=datetime.now()
        )
    
    with col2:
        end_date = st.date_input(
            "Date de fin",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    with col3:
        quartier_filter = st.selectbox(
            "Quartier",
            ["Tous"] + QUARTIERS_DAKAR
        )
    
    # Récupérer l'historique
    df_historique = db.get_recent_predictions(limit=1000)
    
    if not df_historique.empty:
        # Filtrer par date
        df_historique['date_heure'] = pd.to_datetime(df_historique['date_heure'])
        mask = (df_historique['date_heure'].dt.date >= start_date) & (df_historique['date_heure'].dt.date <= end_date)
        df_filtered = df_historique[mask]
        
        # Filtrer par quartier si nécessaire
        if quartier_filter != "Tous":
            df_filtered = df_filtered[df_filtered['quartier'] == quartier_filter]
        
        if not df_filtered.empty:
            # Métriques de la période
            st.subheader("📊 Métriques de la Période")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Prédictions", len(df_filtered))
            
            with col2:
                risque_moyen = df_filtered['risque_global'].mean()
                st.metric("Risque Moyen", f"{risque_moyen:.1f}%")
            
            with col3:
                nb_critique = len(df_filtered[df_filtered['risque_global'] >= SEUILS_RISQUE['eleve']])
                pct_critique = (nb_critique / len(df_filtered)) * 100
                st.metric("Alertes Critiques", f"{nb_critique} ({pct_critique:.1f}%)")
            
            with col4:
                temp_moy = df_filtered['temperature'].mean()
                st.metric("Temp. Moyenne", f"{temp_moy:.1f}°C")
            
            # Graphique d'évolution temporelle
            st.markdown("---")
            st.subheader("📈 Évolution du Risque")
            
            fig_evolution = px.line(
                df_filtered,
                x='date_heure',
                y='risque_global',
                color='quartier' if quartier_filter == "Tous" else None,
                title="Évolution du Risque de Coupure",
                labels={
                    'date_heure': 'Date et Heure',
                    'risque_global': 'Risque (%)',
                    'quartier': 'Quartier'
                }
            )
            
            # Ajouter les seuils
            fig_evolution.add_hline(y=SEUILS_RISQUE['faible'], line_dash="dash", line_color="green", annotation_text="Seuil Faible")
            fig_evolution.add_hline(y=SEUILS_RISQUE['moyen'], line_dash="dash", line_color="orange", annotation_text="Seuil Moyen")
            fig_evolution.add_hline(y=SEUILS_RISQUE['eleve'], line_dash="dash", line_color="red", annotation_text="Seuil Élevé")
            
            st.plotly_chart(fig_evolution, width='stretch')
            
            # Tableau des dernières prédictions
            st.markdown("---")
            st.subheader("📋 Dernières Prédictions")
            
            # Formater le dataframe
            df_display = df_filtered.tail(50).copy()
            df_display['date_heure'] = df_display['date_heure'].dt.strftime('%Y-%m-%d %H:%M')
            df_display['temperature'] = df_display['temperature'].apply(lambda x: f"{x:.1f}°C")
            df_display['humidite'] = df_display['humidite'].apply(lambda x: f"{x:.0f}%")
            df_display['vitesse_vent'] = df_display['vitesse_vent'].apply(lambda x: f"{x:.0f} km/h")
            df_display['consommation'] = df_display['consommation'].apply(lambda x: f"{x:.0f} MW")
            df_display['risque_global'] = df_display['risque_global'].apply(lambda x: f"{x:.1f}%")
            
            df_display = df_display[['date_heure', 'quartier', 'temperature', 'humidite', 'vitesse_vent', 'consommation', 'risque_global']]
            df_display.columns = ['Date/Heure', 'Quartier', 'Temp.', 'Hum.', 'Vent', 'Conso.', 'Risque']
            
            st.dataframe(df_display, width='stretch', hide_index=True)
            
            # Heatmap des risques par jour et quartier
            if quartier_filter == "Tous" and len(df_filtered) > 10:
                st.markdown("---")
                st.subheader("🗓️ Heatmap des Risques")
                
                # Préparer les données
                df_filtered['date'] = df_filtered['date_heure'].dt.date
                heatmap_data = df_filtered.groupby(['date', 'quartier'])['risque_global'].mean().reset_index()
                heatmap_pivot = heatmap_data.pivot(index='quartier', columns='date', values='risque_global')
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=heatmap_pivot.values,
                    x=[str(d) for d in heatmap_pivot.columns],
                    y=heatmap_pivot.index,
                    colorscale='RdYlGn_r',
                    colorbar=dict(title="Risque (%)")
                ))
                
                fig_heatmap.update_layout(
                    title="Risque Moyen par Jour et Quartier",
                    xaxis_title="Date",
                    yaxis_title="Quartier"
                )
                
                st.plotly_chart(fig_heatmap, width='stretch')
        
        else:
            st.info("📊 Aucune donnée pour la période sélectionnée")
    
    else:
        st.info("📊 Aucun historique disponible. Lancez quelques prédictions pour générer des données !")

# ================================================================
# FOOTER
# ================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>⚡ Dakar Power Prediction</strong> | Développé avec Streamlit, LightGBM, LSTM & Supabase</p>
    <p>📊 Données stockées dans Supabase (PostgreSQL) | 🏘️ Quartiers: 6 | 🤖 Modèles: LightGBM + LSTM</p>
</div>
""", unsafe_allow_html=True)