# Fichier : src/database.py
# Gestion de la base de données (MySQL / SQLite)
# ================================================
#
# Ce fichier gère toutes les interactions avec la base de données.
# Pourquoi une base de données ? Pour stocker :
# 1. Les données historiques (52,704 enregistrements)
# 2. Les prédictions effectuées par l'application Streamlit
#
# Technologies utilisées :
# - SQLAlchemy : ORM (Object-Relational Mapping) Python
#   → Permet d'écrire du code Python au lieu de SQL brut
#   → Compatible MySQL et SQLite (même code pour les deux)
#
# Structure :
# 1. Modèles ORM (définition des tables)
# 2. Classe DatabaseManager (gestion de la connexion et opérations)
# 3. Fonctions utilitaires (initialisation, import CSV)

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# SQLAlchemy pour la gestion des bases de données
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, 
    DateTime, Boolean, Text, MetaData, Table, inspect, text
)
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

# Import de la configuration
try:
    from src.config import (
        DATABASE_TYPE, SQLITE_DB_FILE, MYSQL_CONFIG,
        get_db_connection_string, MESSAGES
    )
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from src.config import (
        DATABASE_TYPE, SQLITE_DB_FILE, MYSQL_CONFIG,
        get_db_connection_string, MESSAGES
    )

# Base pour les modèles ORM
# Toutes les classes de tables héritent de cette base
Base = declarative_base()


# ====================================
# 1. MODÈLES ORM (DÉFINITION DES TABLES)
# ====================================
#
# Les classes ci-dessous définissent la structure des tables SQL.
# SQLAlchemy les convertira automatiquement en commandes CREATE TABLE.

class Enregistrement(Base):
    """
    Table des enregistrements de données historiques.
    
    Cette table stocke les 52,704 enregistrements générés (ou les données
    réelles de SENELEC si disponibles).
    
    Colonnes :
        - id : Identifiant unique auto-incrémenté
        - date_heure : Timestamp de l'enregistrement (indexé pour rapidité)
        - quartier : Nom du quartier (indexé pour filtrage rapide)
        - temp_celsius : Température en °C
        - humidite_percent : Humidité relative en %
        - vitesse_vent : Vitesse du vent en km/h
        - conso_megawatt : Consommation électrique en MW
        - coupure : Boolean (True=coupure, False=pas de coupure)
        - created_at : Date d'insertion dans la BD
    
    Index créés :
        - date_heure : Pour les requêtes temporelles (ex: dernières 24h)
        - quartier : Pour filtrer par zone géographique
    
    Taille estimée : ~10 MB pour 52,704 lignes
    """
    __tablename__ = 'enregistrements'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date_heure = Column(DateTime, nullable=False, index=True)
    quartier = Column(String(100), nullable=False, index=True)
    temp_celsius = Column(Float, nullable=False)
    humidite_percent = Column(Float, nullable=False)
    vitesse_vent = Column(Float, nullable=False)
    conso_megawatt = Column(Float, nullable=False)
    coupure = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        """Représentation textuelle pour le debugging"""
        return f"<Enregistrement(id={self.id}, quartier='{self.quartier}', date='{self.date_heure}')>"


class Prediction(Base):
    """
    Table des prédictions effectuées par l'application Streamlit.
    
    Chaque fois qu'un utilisateur fait une prédiction dans l'interface,
    on sauvegarde les paramètres et les résultats ici.
    
    Utilité :
        - Audit : Traçabilité de toutes les prédictions
        - Analyse : Comparer prédictions vs réalité
        - Statistiques : Quels quartiers sont les plus consultés ?
    
    Colonnes :
        Inputs :
            - date_heure, quartier, temp_celsius, humidite_percent,
              vitesse_vent, conso_megawatt
        
        Outputs :
            - proba_lgbm : Probabilité selon LightGBM (0.0-1.0)
            - proba_lstm : Probabilité selon LSTM (0.0-1.0)
            - proba_moyenne : Moyenne des deux (0.0-1.0)
            - prediction : Décision binaire (0 ou 1)
        
        Metadata :
            - modele_utilise : 'lgbm', 'lstm', ou 'ensemble'
            - seuil_decision : Seuil utilisé (ex: 0.21)
            - created_at : Timestamp de la prédiction
    """
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date_heure = Column(DateTime, nullable=False, index=True)
    quartier = Column(String(100), nullable=False, index=True)
    
    # Conditions d'entrée (ce que l'utilisateur a saisi)
    temp_celsius = Column(Float, nullable=False)
    humidite_percent = Column(Float, nullable=False)
    vitesse_vent = Column(Float, nullable=False)
    conso_megawatt = Column(Float, nullable=False)
    
    # Prédictions (résultats des modèles)
    proba_lgbm = Column(Float, nullable=False)
    proba_lstm = Column(Float, nullable=False)
    proba_moyenne = Column(Float, nullable=False)
    prediction = Column(Boolean, nullable=False)  # 0 ou 1
    
    # Métadonnées (pour audit et analyse)
    modele_utilise = Column(String(50), default='ensemble')  # lgbm, lstm, ensemble
    seuil_decision = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        """Représentation textuelle pour le debugging"""
        return f"<Prediction(id={self.id}, quartier='{self.quartier}', proba={self.proba_moyenne:.2%})>"


# ====================================
# 2. CLASSE DE GESTION DE LA BASE DE DONNÉES
# ====================================

class DatabaseManager:
    """
    Gestionnaire centralisé pour toutes les opérations sur la base de données.
    
    Cette classe encapsule :
    - La connexion à la BD (MySQL ou SQLite)
    - La création/suppression de tables
    - L'insertion de données
    - Les requêtes SELECT avec filtres
    - Les statistiques
    
    Avantages de cette approche :
    - Code réutilisable (pas de SQL dupliqué partout)
    - Gestion d'erreurs centralisée
    - Facile à tester
    - Compatible MySQL ET SQLite (même code)
    
    Exemple d'utilisation :
        db = DatabaseManager()
        db.connect()
        db.create_tables()
        db.insert_raw_data(df)
        stats = db.get_statistics()
        db.close()
    """
    
    def __init__(self, db_type=DATABASE_TYPE):
        """
        Initialise le gestionnaire (sans se connecter encore).
        
        Args:
            db_type (str): Type de BD ('sqlite' ou 'mysql')
                          Par défaut, utilise config.DATABASE_TYPE
        
        Note :
            La connexion n'est PAS établie dans __init__ pour éviter
            les erreurs si la BD n'est pas disponible. On appelle
            explicitement connect() après.
        """
        self.db_type = db_type
        self.engine = None  # Moteur SQLAlchemy (connexion)
        self.Session = None  # Session factory (pour transactions)
        self.metadata = MetaData()  # Métadonnées des tables
        
    def connect(self):
        """
        Établit la connexion à la base de données.
        
        Returns:
            bool: True si succès, False si échec
        
        Processus :
            1. Récupérer la chaîne de connexion (depuis config.py)
            2. Créer le moteur SQLAlchemy
            3. Créer la session factory
            4. Tester la connexion avec SELECT 1
        
        Chaînes de connexion :
            SQLite : sqlite:///data/dakar_power.db
            MySQL : mysql+pymysql://user:password@localhost:3306/dakar_predictions
        """
        try:
            # Récupérer la chaîne de connexion depuis config.py
            connection_string = get_db_connection_string()
            print(f"🔗 Connexion à la base de données ({self.db_type})...")
            
            # Créer le moteur SQLAlchemy
            # echo=False : Pas d'affichage des requêtes SQL (mettre True pour debug)
            # pool_pre_ping=True : Vérifier que la connexion est vivante avant usage
            self.engine = create_engine(
                connection_string,
                echo=False,
                pool_pre_ping=True
            )
            
            # Créer la session factory (pour les transactions ORM)
            self.Session = sessionmaker(bind=self.engine)
            
            # Tester la connexion avec une requête simple
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print(f"   ✅ Connexion établie avec succès !")
            return True
            
        except SQLAlchemyError as e:
            # Erreur de connexion (serveur MySQL down, mot de passe incorrect, etc.)
            print(f"   ❌ Erreur de connexion : {e}")
            return False
    
    def create_tables(self):
        """
        Crée toutes les tables définies dans les modèles ORM.
        
        Returns:
            bool: True si succès, False si échec
        
        Cette méthode génère automatiquement les commandes SQL CREATE TABLE
        à partir des classes Enregistrement et Prediction.
        
        Si les tables existent déjà, SQLAlchemy ne fait rien (pas d'erreur).
        
        Exemple de SQL généré pour Enregistrement :
            CREATE TABLE enregistrements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_heure DATETIME NOT NULL,
                quartier VARCHAR(100) NOT NULL,
                ...
                INDEX idx_date_heure (date_heure),
                INDEX idx_quartier (quartier)
            );
        """
        try:
            print("🏗️ Création des tables...")
            
            # Créer toutes les tables définies dans Base
            # create_all() génère les CREATE TABLE pour chaque classe
            Base.metadata.create_all(self.engine)
            
            # Vérifier quelles tables ont été créées
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            print(f"   ✅ Tables créées : {', '.join(tables)}")
            return True
            
        except SQLAlchemyError as e:
            print(f"   ❌ Erreur lors de la création des tables : {e}")
            return False
    
    def drop_tables(self):
        """
        Supprime TOUTES les tables de la base de données.
        
        ⚠️ ATTENTION : Cette opération est IRRÉVERSIBLE !
        Toutes les données seront perdues.
        
        Returns:
            bool: True si succès, False si échec
        
        Utilisation typique : Réinitialisation complète de la BD
        (par exemple, après avoir modifié la structure des tables)
        """
        try:
            print("⚠️ Suppression de toutes les tables...")
            Base.metadata.drop_all(self.engine)
            print("   ✅ Tables supprimées")
            return True
        except SQLAlchemyError as e:
            print(f"   ❌ Erreur : {e}")
            return False
    
    def insert_raw_data(self, df):
        """
        Insère les données brutes (raw_data.csv) dans la table enregistrements.
        
        Args:
            df (pd.DataFrame): DataFrame avec colonnes :
                              date_heure, quartier, temp_celsius, humidite_percent,
                              vitesse_vent, conso_megawatt, coupure
            
        Returns:
            int: Nombre de lignes insérées (0 si échec)
        
        Méthode d'insertion :
            pandas.to_sql() avec method='multi' et chunksize=1000
            → Insertion par lots de 1000 lignes (rapide et efficace)
        
        Exemple d'utilisation :
            df = pd.read_csv('data/raw/raw_data.csv')
            db.insert_raw_data(df)
        """
        try:
            print(f"💾 Insertion de {len(df)} enregistrements...")
            
            # Préparer les données
            df_insert = df.copy()
            df_insert['created_at'] = datetime.now()  # Timestamp d'insertion
            
            # Mapping des colonnes (au cas où elles auraient des noms différents)
            column_mapping = {
                'date_heure': 'date_heure',
                'quartier': 'quartier',
                'temp_celsius': 'temp_celsius',
                'humidite_percent': 'humidite_percent',
                'vitesse_vent': 'vitesse_vent',
                'conso_megawatt': 'conso_megawatt',
                'coupure': 'coupure'
            }
            
            # Sélectionner uniquement les colonnes nécessaires
            cols_to_insert = [col for col in column_mapping.keys() if col in df_insert.columns]
            df_to_insert = df_insert[cols_to_insert + ['created_at']]
            
            # Insérer dans la BD avec pandas.to_sql()
            # if_exists='append' : Ajouter aux données existantes
            # index=False : Ne pas insérer l'index pandas
            # method='multi' : Insertion par lots (rapide)
            # chunksize=1000 : 1000 lignes par batch
            df_to_insert.to_sql(
                'enregistrements',
                self.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )
            
            print(f"   ✅ {len(df)} enregistrements insérés")
            return len(df)
            
        except SQLAlchemyError as e:
            print(f"   ❌ Erreur d'insertion : {e}")
            return 0
    
    def insert_prediction(self, prediction_data):
        """
        Insère UNE prédiction dans la table predictions.
        
        Args:
            prediction_data (dict): Dictionnaire contenant :
                - date_heure, quartier
                - temp_celsius, humidite_percent, vitesse_vent, conso_megawatt
                - proba_lgbm, proba_lstm, proba_moyenne, prediction
                - modele_utilise (optionnel), seuil_decision (optionnel)
            
        Returns:
            int: ID de la prédiction insérée (ou None si échec)
        
        Cette méthode est appelée par l'application Streamlit chaque fois
        qu'un utilisateur fait une prédiction.
        
        Exemple d'utilisation (dans Streamlit) :
            pred_data = {
                'date_heure': datetime.now(),
                'quartier': 'Guediawaye',
                'temp_celsius': 35.0,
                'humidite_percent': 70.0,
                'vitesse_vent': 15.0,
                'conso_megawatt': 900.0,
                'proba_lgbm': 0.245,
                'proba_lstm': 0.507,
                'proba_moyenne': 0.376,
                'prediction': 1,
                'modele_utilise': 'ensemble',
                'seuil_decision': 0.21
            }
            pred_id = db.insert_prediction(pred_data)
        """
        try:
            # Créer une session (transaction)
            session = self.Session()
            
            # Créer un objet Prediction (modèle ORM)
            prediction = Prediction(
                date_heure=prediction_data['date_heure'],
                quartier=prediction_data['quartier'],
                temp_celsius=prediction_data['temp_celsius'],
                humidite_percent=prediction_data['humidite_percent'],
                vitesse_vent=prediction_data['vitesse_vent'],
                conso_megawatt=prediction_data['conso_megawatt'],
                proba_lgbm=prediction_data['proba_lgbm'],
                proba_lstm=prediction_data['proba_lstm'],
                proba_moyenne=prediction_data['proba_moyenne'],
                prediction=prediction_data['prediction'],
                modele_utilise=prediction_data.get('modele_utilise', 'ensemble'),
                seuil_decision=prediction_data.get('seuil_decision', 0.5)
            )
            
            # Ajouter à la session et commiter
            session.add(prediction)
            session.commit()
            
            # Récupérer l'ID auto-généré
            pred_id = prediction.id
            session.close()
            
            return pred_id
            
        except SQLAlchemyError as e:
            print(f"❌ Erreur d'insertion de prédiction : {e}")
            session.rollback()  # Annuler la transaction en cas d'erreur
            session.close()
            return None
    
    def get_enregistrements(self, quartier=None, date_debut=None, date_fin=None, limit=1000):
        """
        Récupère les enregistrements historiques avec filtres optionnels.
        
        Args:
            quartier (str): Filtrer par quartier (ex: 'Guediawaye')
            date_debut (datetime): Date de début (ex: datetime(2024, 11, 1))
            date_fin (datetime): Date de fin
            limit (int): Nombre max de résultats (défaut: 1000)
            
        Returns:
            pd.DataFrame: DataFrame avec les enregistrements
        
        Exemples d'utilisation :
            # Toutes les données de Guediawaye (max 1000)
            df = db.get_enregistrements(quartier='Guediawaye')
            
            # Dernières 24h tous quartiers confondus
            df = db.get_enregistrements(
                date_debut=datetime.now() - timedelta(hours=24),
                limit=500
            )
            
            # Novembre 2024 pour Yoff
            df = db.get_enregistrements(
                quartier='Yoff',
                date_debut=datetime(2024, 11, 1),
                date_fin=datetime(2024, 11, 30)
            )
        """
        try:
            # Construction de la requête SQL avec filtres dynamiques
            query = f"SELECT * FROM enregistrements WHERE 1=1"
            
            if quartier:
                query += f" AND quartier = '{quartier}'"
            if date_debut:
                query += f" AND date_heure >= '{date_debut}'"
            if date_fin:
                query += f" AND date_heure <= '{date_fin}'"
            
            # Tri par date décroissante (plus récent en premier)
            query += f" ORDER BY date_heure DESC LIMIT {limit}"
            
            # Exécuter la requête et retourner un DataFrame
            df = pd.read_sql(query, self.engine)
            return df
            
        except SQLAlchemyError as e:
            print(f"❌ Erreur de récupération : {e}")
            return pd.DataFrame()  # Retourner un DataFrame vide en cas d'erreur
    
    def get_predictions(self, quartier=None, date_debut=None, date_fin=None, limit=100):
        """
        Récupère les prédictions effectuées avec filtres optionnels.
        
        Args:
            quartier (str): Filtrer par quartier
            date_debut (datetime): Date de début
            date_fin (datetime): Date de fin
            limit (int): Nombre max de résultats (défaut: 100)
            
        Returns:
            pd.DataFrame: DataFrame avec les prédictions
        
        Utilité :
            - Analyser les prédictions passées
            - Comparer prédictions vs réalité (si on a les vraies coupures)
            - Statistiques sur l'utilisation de l'application
        """
        try:
            query = f"SELECT * FROM predictions WHERE 1=1"
            
            if quartier:
                query += f" AND quartier = '{quartier}'"
            if date_debut:
                query += f" AND date_heure >= '{date_debut}'"
            if date_fin:
                query += f" AND date_heure <= '{date_fin}'"
            
            # Tri par date de création (created_at) décroissante
            query += f" ORDER BY created_at DESC LIMIT {limit}"
            
            df = pd.read_sql(query, self.engine)
            return df
            
        except SQLAlchemyError as e:
            print(f"❌ Erreur de récupération : {e}")
            return pd.DataFrame()
    
    def get_statistics(self):
        """
        Récupère des statistiques générales sur la base de données.
        
        Returns:
            dict: Dictionnaire avec les statistiques :
                - total_enregistrements : Nombre total de lignes
                - total_coupures : Nombre total de coupures
                - total_predictions : Nombre de prédictions effectuées
                - quartiers : Liste des quartiers
                - periode_debut, periode_fin : Période couverte
        
        Utilité :
            - Afficher un dashboard de statistiques
            - Vérifier que les données sont bien chargées
            - Monitoring de l'application
        """
        try:
            stats = {}
            
            # Nombre total d'enregistrements
            query = "SELECT COUNT(*) as total FROM enregistrements"
            result = pd.read_sql(query, self.engine)
            stats['total_enregistrements'] = int(result['total'].iloc[0])
            
            # Nombre total de coupures
            query = "SELECT COUNT(*) as total FROM enregistrements WHERE coupure = 1"
            result = pd.read_sql(query, self.engine)
            stats['total_coupures'] = int(result['total'].iloc[0])
            
            # Nombre de prédictions effectuées
            query = "SELECT COUNT(*) as total FROM predictions"
            result = pd.read_sql(query, self.engine)
            stats['total_predictions'] = int(result['total'].iloc[0])
            
            # Liste des quartiers uniques
            query = "SELECT DISTINCT quartier FROM enregistrements"
            result = pd.read_sql(query, self.engine)
            stats['quartiers'] = result['quartier'].tolist()
            
            # Période couverte par les données
            query = "SELECT MIN(date_heure) as debut, MAX(date_heure) as fin FROM enregistrements"
            result = pd.read_sql(query, self.engine)
            stats['periode_debut'] = result['debut'].iloc[0]
            stats['periode_fin'] = result['fin'].iloc[0]
            
            return stats
            
        except SQLAlchemyError as e:
            print(f"❌ Erreur de récupération des stats : {e}")
            return {}
    
    def close(self):
        """
        Ferme proprement la connexion à la base de données.
        
        Libère les ressources (connexions au pool, mémoire).
        Toujours appeler cette méthode à la fin !
        
        Exemple :
            try:
                db = DatabaseManager()
                db.connect()
                # ... opérations ...
            finally:
                db.close()  # Même en cas d'erreur
        """
        if self.engine:
            self.engine.dispose()
            print("🔌 Connexion fermée")


# ====================================
# 3. FONCTIONS UTILITAIRES
# ====================================

def init_database(drop_existing=False):
    """
    Initialise la base de données de manière complète.
    
    Args:
        drop_existing (bool): Si True, supprime et recrée les tables
                             (⚠️ PERTE DE DONNÉES !)
        
    Returns:
        DatabaseManager: Instance du gestionnaire (ou None si échec)
    
    Cette fonction est pratique pour démarrer rapidement :
    - Connexion
    - Création des tables
    - Gestion d'erreurs
    
    Exemple d'utilisation (dans un script) :
        db = init_database(drop_existing=True)  # Reset complet
        if db:
            import_csv_to_db('data/raw/raw_data.csv', db)
            db.close()
    """
    print("\n" + "="*60)
    print("🗄️ INITIALISATION DE LA BASE DE DONNÉES")
    print("="*60 + "\n")
    
    db = DatabaseManager()
    
    # Connexion
    if not db.connect():
        return None
    
    # Supprimer les tables existantes si demandé
    if drop_existing:
        db.drop_tables()
    
    # Créer les tables
    db.create_tables()
    
    print("\n" + "="*60)
    print("✅ BASE DE DONNÉES INITIALISÉE")
    print("="*60 + "\n")
    
    return db


def import_csv_to_db(csv_file, db_manager):
    """
    Importe un fichier CSV dans la base de données.
    
    Args:
        csv_file (Path): Chemin vers raw_data.csv
        db_manager (DatabaseManager): Instance du gestionnaire
        
    Returns:
        int: Nombre de lignes importées
    
    Utilisation typique :
        db = init_database()
        count = import_csv_to_db('data/raw/raw_data.csv', db)
        print(f"{count} lignes importées")
        db.close()
    """
    print(f"\n📂 Import du fichier : {csv_file}")
    
    # Charger le CSV
    df = pd.read_csv(csv_file, parse_dates=['date_heure'])
    print(f"   📊 {len(df)} lignes chargées")
    
    # Insérer dans la BD
    count = db_manager.insert_raw_data(df)
    
    return count


# ====================================
# 4. FONCTION DE TEST
# ====================================

def main():
    """
    Fonction de test pour vérifier que la BD fonctionne.
    
    Exécutée quand on lance : python src/database.py
    
    Teste :
    - Initialisation de la BD
    - Affichage des statistiques
    - Fermeture propre
    """
    # Initialiser la BD (sans supprimer les données existantes)
    db = init_database(drop_existing=False)
    
    if db:
        # Afficher les statistiques
        print("\n📊 Statistiques de la base de données :")
        stats = db.get_statistics()
        for key, value in stats.items():
            print(f"   • {key}: {value}")
        
        # Fermer la connexion
        db.close()


if __name__ == "__main__":
    main()