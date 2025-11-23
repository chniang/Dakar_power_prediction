⚡ Dakar Power Prediction
Système intelligent de prédiction des coupures d'électricité à Dakar (ML + Deep Learning)
📋 Table des Matières

Aperçu

Fonctionnalités

Démo

Architecture

Installation

Utilisation

Modèles ML

Résultats

Structure du Projet

Technologies

Roadmap

Contribution

Tests

Contact

Ressources Supplémentaires

🎯 Aperçu

Dakar Power Prediction est une application web intelligente qui prédit en temps réel le risque de coupure d’électricité dans 6 quartiers de Dakar.
Le système combine Machine Learning (LightGBM) et Deep Learning (LSTM) pour générer des prédictions fiables et exploitables.

🔥 Problématique

Les coupures d’électricité à Dakar affectent :

👨‍👩‍👧‍👦 Ménages : appareils endommagés, nourriture qui se gâte

🏢 Entreprises : perte de productivité

🏭 Industries : coûts opérationnels élevés

✅ Solution

Une plateforme web accessible 24/7 permettant d’anticiper les coupures pour mieux s’y préparer.

✨ Fonctionnalités
🎯 Prédiction Immédiate

Temps réel (<1 seconde)

Sélection du quartier

Ajustement manuel des paramètres

Jauge visuelle (vert/orange/rouge)

🗺️ Carte Interactive

Affichage géographique des 6 quartiers

Marqueurs colorés selon le risque

Mise à jour automatique

📊 Analyse par Quartier

Statistiques détaillées

Graphiques comparatifs

Taux de coupures historiques

📈 Historique & Tendances

Visualisations temporelles 7 jours

Courbes consommation vs température

Coupures réelles marquées

🚀 Démo
🔗 Application Web

(À venir)

📸 Screenshots
<details> <summary>Cliquez pour afficher</summary>

Prédiction Immédiate

Carte Interactive

Analyse par Quartier

Historique

</details>
🏗️ Architecture
graph LR
    A[Données<br/>Synthétiques] --> B[Preprocessing<br/>& Features]
    B --> C[Entraînement<br/>LightGBM + LSTM]
    C --> D[Modèles<br/>Entraînés]
    D --> E[Interface<br/>Streamlit]
    E --> F[Utilisateur<br/>Final]
    G[Base de<br/>Données MySQL] --> E

🔄 Pipeline de Données

52 560 observations (1 an × 6 quartiers)

9 features générées

Entraînement :

LightGBM → 2 min

LSTM → 8 min

Déploiement : Streamlit

📦 Installation
🔧 Prérequis

Python 3.12+

pip

Git

MySQL 8.x (ou Docker)

🛠️ Étapes
# 1. Cloner le repository
git clone https://github.com/votre-username/dakar-power-prediction.git
cd dakar-power-prediction

# 2. Créer l'environnement virtuel
python -m venv venv

# 3. Activer l'environnement
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Générer les données
python scripts/1_generate_data.py --import-db

# 6. Entraîner les modèles
python scripts/2_train_models.py

# 7. Lancer Streamlit
streamlit run streamlit_app/app.py

🚀 Installation Rapide avec Docker
docker build -t dakar-power-prediction .
docker run -p 8501:8501 dakar-power-prediction

🎮 Utilisation
Mode Dev
streamlit run streamlit_app/app.py --server.runOnSave true

Mode Production
streamlit run streamlit_app/app.py \
  --server.port 80 \
  --server.headless true \
  --browser.gatherUsageStats false

🧩 Exemple de prédiction (Python)
from streamlit_app.utils import load_models, make_prediction_single

# Charger les modèles
lgbm, threshold_lgbm, lstm, threshold_lstm, scaler, encoder = load_models()

# Données d'entrée
input_data = {
    'temperature': 35.0,
    'humidite': 80.0,
    'vent': 25.0,
    'consommation': 1200.0
}

# Prédiction
result = make_prediction_single(
    input_data, 
    'Guédiawaye',
    lgbm, threshold_lgbm,
    lstm, threshold_lstm,
    scaler, encoder
)

print(f"Probabilité : {result['proba_moyenne']*100:.2f}%")
print(f"Statut : {result['statut']}")

🤖 Modèles ML
⭐ LightGBM (modèle principal)

Rapide (<100 ms)

Pas besoin d’historique

Accuracy : 74.72%

ROC-AUC : 65.94%

🧠 LSTM (Deep Learning)

Capture les séquences

Architecture multi-couches

Accuracy : 76.14%

🔗 Ensemble Learning

Combinaison LGBM + LSTM pour une meilleure robustesse.

📊 Résultats
🔥 Comparaison
Métrique	LightGBM	LSTM	Meilleur
Accuracy	74.72%	76.14%	LSTM
Precision	12.70%	9.95%	LightGBM
Recall	44.13%	29.69%	LightGBM
F1-Score	19.72%	14.91%	LightGBM
ROC-AUC	65.94%	55.55%	LightGBM
🔥 Matrice de Confusion (LightGBM)
                   Prédictions
               Pas Coupure | Coupure
Réel
Pas Coupure      TN: 7234   FP: 607
Coupure          FN: 318    TP: 251

🎯 Importance des Features

Consommation (35%)

Quartier (25%)

Température (19%)

Heure (13%)

Heure de pointe (8%)

📁 Structure du Projet
dakar_power_prediction/
├── data/
│   ├── processed/
│   ├── raw/
│   └── dakar_power.db
├── evaluation_results/
├── models/
├── screenshots/
├── scripts/
├── src/
├── streamlit_app/
├── requirements.txt
├── README.md
└── .gitignore

🛠️ Technologies
Langage

Python 3.12

Frontend

Streamlit

Plotly

Machine Learning

LightGBM

TensorFlow

scikit-learn

SMOTE (imbalanced-learn)

Data

Pandas

NumPy

MySQL

Déploiement

Docker

Streamlit Cloud

🗺️ Roadmap
✅ Phase 1 — MVP (terminée)

Pipeline complet

2 modèles ML

Interface Streamlit

Documentation

🔄 Phase 2 — Amélioration (en cours)

Déploiement Streamlit Cloud

Données réelles SENELEC

Optimisation hyperparamètres

📅 Phase 3 — Extension (2026)

20+ quartiers

API REST

Alerts SMS/Email

📱 Phase 4 — Mobile (2026)

App iOS & Android

Notifications push

🤝 Phase 5 — Partenariats

SENELEC

Dashboard Pro

🤝 Contribution

Fork le repo

Créer une branche

git checkout -b feature/NouvelleFeature


Commit

git commit -m "Add: NouvelleFeature"


Push

git push origin feature/NouvelleFeature


Ouvrir une Pull Request

Règles

Tests unitaires

PEP8

Docstrings

PR détaillée

🧪 Tests
pip install pytest pytest-cov
pytest --cov=src --cov-report=html

📧 Contact

Développeur : Cheikh Niang
📩 Email : cheikhniang159@gmail.com

🔗 LinkedIn : https://www.linkedin.com/in/cheikh-niang-5370091b5/

💻 GitHub : https://github.com/chniang

📦 Projet : https://github.com/chniang/Dakar_power_prediction

📚 Ressources Supplémentaires

Documentation technique complète (PDF)

Application déployée (à venir)

<div align="center">
Développé avec ❤️ à Dakar, Sénégal

⚡ Anticiper pour mieux préparer ⚡
⬆ Retour en haut

</div>
