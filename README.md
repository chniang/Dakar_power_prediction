⚡ Dakar Power PredictionSystème intelligent de prédiction des coupures d'électricité à Dakar utilisant Machine Learning et Deep Learning📋 Table des MatièresAperçuFonctionnalitésDémoArchitectureInstallationUtilisationModèles MLRésultatsStructure du ProjetTechnologiesRoadmapContributionContact🎯 AperçuDakar Power Prediction est une application web intelligente qui prédit en temps réel le risque de coupure d'électricité pour 6 quartiers de Dakar. Le système combine deux modèles de Machine Learning (LightGBM et LSTM) pour fournir des prédictions fiables et exploitables.ProblématiqueLes coupures d'électricité à Dakar impactent négativement :👨‍👩‍👧‍👦 Ménages : Équipements endommagés, alimentation gâchée🏢 Entreprises : Productivité perdue, données non sauvegardées🏭 Industrie : Coûts opérationnels élevésSolutionUne plateforme web accessible 24/7 qui permet d'anticiper les coupures pour mieux s'y préparer.✨ Fonctionnalités🎯 Prédiction ImmédiatePrédiction en temps réel (< 1 seconde)Sélection du quartierAjustement des paramètres météo et consommationAffichage du niveau de risque (Faible/Modéré/Élevé)Jauge visuelle colorée🗺️ Carte InteractiveVisualisation géographique des 6 quartiersMarqueurs colorés selon le niveau de risqueMise à jour automatique en temps réelTableau récapitulatif📊 Analyse par QuartierStatistiques historiquesGraphiques comparatifsTaux de coupures par quartier📈 Historique & TendancesGraphiques temporels (7 jours)Consommation et températureMarqueurs de coupures réelles🚀 DémoApplication Web🔗 [Lien vers l'application déployée] (à venir)Screenshots<details><summary>📸 Cliquez pour voir les captures d'écran</summary>Prédiction ImmédiateCarte InteractiveAnalyse par QuartierHistorique</details>🏗️ ArchitectureExtrait de codegraph LR
    A[Données<br/>Synthétiques] --> B[Preprocessing<br/>& Features]
    B --> C[Entraînement<br/>LightGBM + LSTM]
    C --> D[Modèles<br/>Entraînés]
    D --> E[Interface<br/>Streamlit]
    E --> F[Utilisateur<br/>Final]
    G[Base de<br/>Données MySQL] --> E
Pipeline de DonnéesGénération : 52,560 observations (1 an × 6 quartiers)Feature Engineering : 9 colonnes crééesEntraînement : LightGBM (2 min) + LSTM (8 min)Déploiement : Interface Streamlit interactive📦 InstallationPrérequisPython 3.12+pipGitServeur MySQL (ou Docker pour MySQL)ÉtapesBash# 1. Cloner le repository
git clone https://github.com/votre-username/dakar-power-prediction.git
cd dakar-power-prediction

# 2. Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer et générer les données (Assurez-vous que votre serveur MySQL est démarré)
python scripts/1_generate_data.py --import-db

# 5. Entraîner les modèles
python scripts/2_train_models.py

# 6. Lancer l'application
streamlit run streamlit_app/app.py
Installation Rapide (Docker)Bash# Construire l'image
docker build -t dakar-power-prediction .

# Lancer le conteneur (NOTE: Vous devrez lier ce conteneur à votre conteneur MySQL)
docker run -p 8501:8501 dakar-power-prediction
Accédez à l'application sur http://localhost:8501🎮 UtilisationMode DéveloppementBash# Lancer avec auto-reload
streamlit run streamlit_app/app.py --server.runOnSave true
Mode ProductionBash# Avec configuration serveur
streamlit run streamlit_app/app.py \
  --server.port 80 \
  --server.headless true \
  --browser.gatherUsageStats false
Exemple de Prédiction ProgrammatiquePythonfrom streamlit_app.utils import load_models, make_prediction_single

# Charger les modèles
lgbm, threshold_lgbm, lstm, threshold_lstm, scaler, encoder = load_models()

# Données d'entrée
input_data = {
    'temperature': 35.0,  # °C
    'humidite': 80.0,     # %
    'vent': 25.0,         # km/h
    'consommation': 1200.0 # MW
}

# Faire la prédiction
result = make_prediction_single(
    input_data, 
    'Guédiawaye',
    lgbm, threshold_lgbm,
    lstm, threshold_lstm,
    scaler, encoder
)

print(f"Probabilité de coupure : {result['proba_moyenne']*100:.2f}%")
print(f"Statut : {result['statut']}")
🤖 Modèles MLLightGBM (Modèle Principal) ⭐Type : Gradient Boosting  Avantages :Rapide (< 100ms par prédiction)Fonctionne sans historiqueMeilleur équilibre Precision/RecallPerformances :Accuracy : 74.72%Precision : 12.70%Recall : 44.13%F1-Score : 19.72% ⭐ROC-AUC : 65.94%LSTM (Réseau de Neurones)Type : Long Short-Term Memory  Avantages :Capture les tendances temporellesUtile pour prédictions à moyen termeArchitecture :Input (12 timesteps, 9 features)
  ↓
LSTM (100 units) + Dropout(0.4)
  ↓
LSTM (50 units) + Dropout(0.4)
  ↓
Dense (32) + Dropout(0.3)
  ↓
Dense (16) + Dropout(0.2)
  ↓
Output (1, sigmoid)
Performances :Accuracy : 76.14%F1-Score : 14.91%ROC-AUC : 55.55%Ensemble LearningMoyenne des probabilités des deux modèles pour plus de robustesse.📊 RésultatsComparaison des ModèlesMétriqueLightGBMLSTMMeilleurAccuracy74.72%76.14%LSTMPrecision12.70%9.95%LightGBMRecall44.13%29.69%LightGBMF1-Score19.72%14.91%LightGBM ⭐ROC-AUC65.94%55.55%LightGBMRecommandation : ✅ LightGBM choisi comme modèle principalMatrice de Confusion (LightGBM)                Prédictions
            Pas Coupure  Coupure
          ┌─────────────┬────────┐
Réel      │             │        │
Pas Coup. │ TN: 7,234   │ FP: 607│
          ├─────────────┼────────┤
Coupure   │ FN: 318     │ TP: 251│
          └─────────────┴────────┘
Importance des FeaturesConsommation (35%) - Plus fort prédicteurQuartier (25%) - Zones à risqueTempérature (19%) - Chaleur = risqueHeure (13%) - Heures de pointeHeure de pointe (8%) - Surcharge📁 Structure du Projetdakar_power_prediction/
├── data/
│   ├── processed/
│   │   └── processed_data.csv
│   ├── raw/
│   │   └── raw_data.csv
│   └── dakar_power.db
│
├── evaluation_results/
│   ├── confusion_matrices.png
│   └── evaluation_report_20251118_123625.txt
│
├── models/
│   ├── encoders.pkl
│   ├── lgbm_model.pkl
│   ├── lstm_model.keras
│   ├── lstm_threshold.txt
│   └── scaler.pkl
│
├── screenshots/
│   ├── analyse_par_quartier.png
│   ├── carte_des_risques.png
│   ├── historique_des_tendences.png
│   └── prediction_immediate.png
│
├── scripts/
│   ├── 1_generate_data.py
│   ├── 2_train_models.py
│   └── 3_evaluate_models.py
│
├── src/
│   ├── config.py             
│   ├── database.py
│   ├── data_generator.py
│   ├── data_pipeline.py
│   ├── data_pipeline.py.backup
│   ├── model_trainer.py
│   ├── model_trainer.py.backup
│   └── __init__.py
│
├── streamlit_app/
│   ├── pages/
│   ├── app.py
│   ├── config.py             
│   └── utils.py
│
├── .gitignore
├── README.md
├── requirements.txt
├── 📊 RAPPORT FINAL DE PROJET.md
└── 📘 DOCUMENTATION CONCISE - DAKAR POWER PREDICTION.pdf    
🛠️ TechnologiesLangage & FrameworksPython 3.12Streamlit 1.40.2 - Interface webPlotly 5.24.1 - VisualisationsMachine LearningLightGBM 4.5.0 - Gradient BoostingTensorFlow 2.18.0 - Deep Learningscikit-learn 1.5.2 - Preprocessingimbalanced-learn 0.12.4 - SMOTEData ProcessingPandas 2.2.3 - Manipulation donnéesNumPy 2.1.3 - Calculs numériquesBase de DonnéesMySQL 8.x - StockageDéploiementDocker - ConteneurisationStreamlit Cloud - Hébergement🗺️ Roadmap✅ Phase 1 - MVP (Complétée)[x] Pipeline de données complet[x] 2 modèles ML entraînés[x] Interface Streamlit 4 onglets[x] Documentation complète🔄 Phase 2 - Amélioration (En cours)[ ] Déploiement Streamlit Cloud[ ] Collecte données réelles SENELEC[ ] Optimisation hyperparamètres[ ] Tests unitaires (coverage 80%+)📅 Phase 3 - Extension (Q1 2026)[ ] Extension à 20+ quartiers[ ] Système d'alertes (email, SMS)[ ] API REST[ ] Monitoring en production🚀 Phase 4 - Mobile (Q2 2026)[ ] Application iOS[ ] Application Android[ ] Notifications push[ ] Mode hors-ligne🤝 Phase 5 - Partenariat (Q3 2026)[ ] Partenariat SENELEC[ ] Intégration données temps réel[ ] Prédictions 24h-72h[ ] Dashboard administrateur🤝 ContributionLes contributions sont les bienvenues ! Voici comment participer :1. Fork le ProjetBash# Cloner votre fork
git clone https://github.com/votre-username/dakar-power-prediction.git
2. Créer une BrancheBash# Créer une branche pour votre feature
git checkout -b feature/AmazingFeature
3. Commit vos ChangementsBash# Commit avec message descriptif
git commit -m 'Add: AmazingFeature'
4. Push vers la BrancheBashgit push origin feature/AmazingFeature
5. Ouvrir une Pull RequestOuvrez une PR sur GitHub avec une description détaillée.Règles de Contribution✅ Code documenté (docstrings)✅ Tests unitaires (pytest)✅ Respect PEP 8 (flake8)✅ Commit messages clairs✅ PR avec description détaillée🧪 TestsBash# Installer les dépendances de test
pip install pytest pytest-cov

# Lancer tous les tests
pytest

# Avec coverage
pytest --cov=src --cov-report=html

# Tests spécifiques
pytest tests/test_data_pipeline.py -v
📧 ContactDéveloppeur : Cheikh NiangEmail : cheikhniang159@gmail.comLinkedIn : https://www.linkedin.com/in/cheikh-niang-5370091b5/GitHub : https://github.com/dashboardLien du Projet : https://github.com/chniang/Dakar_power_prediction📚 Ressources Supplémentaires📖 [Documentation Technique Complète](RAPPORT FINAL DE PROJET.md)🌐 [Application Déployée](a venir)⭐ Star History🔖 CitationSi vous utilisez ce projet dans votre recherche, veuillez citer :Extrait de code@software{dakar_power_prediction,
  author = {Votre Nom},
  title = {Dakar Power Prediction: Système de Prédiction des Coupures d'Électricité},
  year = {2025},
  url = {https://github.com/votre-username/dakar-power-prediction}
}
<div align="center">Développé avec ❤️ à Dakar, Sénégal⚡ Anticiper pour mieux préparer ⚡⬆ Retour en haut</div>
