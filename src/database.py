# Fichier : src/database.py
# Gestion de la base de données Supabase (PostgreSQL)
# =========================================================

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from datetime import datetime
import streamlit as st
from src.config import DATABASE_URL, QUARTIERS_DAKAR 

class SupabaseDB:
    """Gestionnaire de base de données Supabase"""
    
    def __init__(self):
        """Initialiser la connexion à Supabase"""
        try:
            # Créer le moteur SQLAlchemy sans pool (pour Streamlit Cloud)
            # DATABASE_URL utilise l'hôte du Pooler de sessions de Supabase
            self.engine = create_engine(
                DATABASE_URL,
                poolclass=NullPool,
                connect_args={
                    "connect_timeout": 10,
                    "sslmode": "require"  # Supabase nécessite SSL
                }
            )
            if self.test_connection():
                # Tente d'initialiser les entrées de la table stats_quartiers si elles sont manquantes
                self._initialize_quartier_stats()
        except Exception as e:
            st.error(f"❌ Erreur connexion Supabase : {e}")
            self.engine = None
    
    def test_connection(self):
        """Tester la connexion"""
        if not self.engine:
            return False
            
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                st.success("✅ Connexion Supabase réussie")
                return True
        except Exception as e:
            # Afficher l'erreur de connexion détaillée si elle se produit
            st.error(f"❌ Test connexion échoué : {e}")
            return False
    
    def save_prediction(self, quartier, temp, hum, vent, conso, pred_lgb, pred_lstm, risque_global):
        """
        Sauvegarder une prédiction dans la table 'predictions' et mettre à jour 'stats_quartiers'.
        """
        if not self.engine:
            return False
        
        try:
            # Insertion dans la table predictions
            query_insert = text("""
                INSERT INTO predictions (
                    quartier, temperature, humidite, vitesse_vent, consommation,
                    prediction_lgb, prediction_lstm, risque_global, date_heure
                ) VALUES (
                    :quartier, :temp, :hum, :vent, :conso,
                    :pred_lgb, :pred_lstm, :risque_global, :date_heure
                )
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query_insert, {
                    'quartier': quartier,
                    'temp': float(temp),
                    'hum': float(hum),
                    'vent': float(vent),
                    'conso': float(conso),
                    'pred_lgb': float(pred_lgb),
                    'pred_lstm': float(pred_lstm),
                    'risque_global': float(risque_global),
                    'date_heure': datetime.now()
                })
                conn.commit()
            
            # Mettre à jour les stats du quartier
            self._update_quartier_stats(quartier, risque_global)
            return True
            
        except Exception as e:
            st.warning(f"⚠️ Erreur sauvegarde (predictions) : {e}")
            return False
    
    def _update_quartier_stats(self, quartier, risque_global):
        """Mettre à jour les statistiques d'un quartier (total_predictions, taux_risque, etc.)"""
        if not self.engine:
            return
            
        try:
            # Requête corrigée pour éviter les conflits dans les calculs UPDATE
            query = text("""
                -- 1. Incrémenter les compteurs
                UPDATE stats_quartiers
                SET 
                    total_predictions = total_predictions + 1,
                    coupures_predites = coupures_predites + CASE WHEN :risque >= 50 THEN 1 ELSE 0 END,
                    derniere_maj = :now
                WHERE quartier = :quartier;
                
                -- 2. Calculer le taux de risque avec les nouvelles valeurs
                UPDATE stats_quartiers
                SET 
                    taux_risque = (coupures_predites::float / total_predictions * 100)
                WHERE quartier = :quartier AND total_predictions > 0;
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query, {
                    'quartier': quartier,
                    'risque': float(risque_global),
                    'now': datetime.now()
                })
                conn.commit()
        except Exception as e:
            # Afficher le warning pour le débogage de l'UPDATE
            st.warning(f"⚠️ Erreur mise à jour stats quartier ({quartier}): {e}") 
            pass  
    
    def _initialize_quartier_stats(self):
        """Initialiser les statistiques des quartiers (si la table est vide)"""
        if not self.engine:
            return

        try:
            # QUARTIERS_DAKAR est une liste (selon src/config.py)
            quartiers_list = QUARTIERS_DAKAR 

            # 1. Récupérer les quartiers déjà présents dans la table
            existing_quartiers_query = text("SELECT quartier FROM stats_quartiers")
            
            with self.engine.connect() as conn:
                existing_quartiers = [row[0] for row in conn.execute(existing_quartiers_query).fetchall()]
                
                # 2. Identifier les quartiers manquants
                quartiers_to_insert = [q for q in quartiers_list if q not in existing_quartiers]
                
                # 3. Insérer les quartiers manquants
                if quartiers_to_insert:
                    insert_query = text("""
                        INSERT INTO stats_quartiers (quartier, total_predictions, coupures_predites, taux_risque, derniere_maj)
                        VALUES (:quartier, 0, 0, 0.0, :now)
                    """)
                    for quartier in quartiers_to_insert:
                        conn.execute(insert_query, {'quartier': quartier, 'now': datetime.now()})
                    conn.commit()
                    st.info(f"✅ Initialisation des stats de {len(quartiers_to_insert)} quartiers effectuée.")

        except Exception as e:
            st.warning(f"⚠️ Erreur initialisation stats quartiers : {e}")

    def get_recent_predictions(self, limit=100):
        """Récupérer les dernières prédictions"""
        if not self.engine:
            return pd.DataFrame()
        
        try:
            query = text("""
                SELECT 
                    quartier,
                    temperature,
                    humidite,
                    vitesse_vent,
                    consommation,
                    prediction_lgb,
                    prediction_lstm,
                    risque_global,
                    date_heure
                FROM predictions
                ORDER BY date_heure DESC
                LIMIT :limit
            """)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={'limit': limit})
            
            return df
            
        except Exception as e:
            st.warning(f"⚠️ Erreur récupération historique : {e}")
            return pd.DataFrame()
    
    def get_quartier_stats(self):
        """Récupérer les statistiques par quartier"""
        if not self.engine:
            return pd.DataFrame()
        
        try:
            query = text("""
                SELECT 
                    quartier,
                    total_predictions,
                    coupures_predites,
                    taux_risque,
                    derniere_maj
                FROM stats_quartiers
                ORDER BY taux_risque DESC
            """)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            return df
            
        except Exception as e:
            st.warning(f"⚠️ Erreur stats quartiers : {e}")
            return pd.DataFrame()
    
    def get_stats_by_date_range(self, start_date, end_date, quartier=None):
        """Récupérer les stats sur une période"""
        if not self.engine:
            return pd.DataFrame()
        
        try:
            # Le reste de vos méthodes de récupération de données...
            if quartier:
                query = text("""
                    SELECT 
                        DATE(date_heure) as date,
                        AVG(risque_global) as risque_moyen,
                        COUNT(*) as nb_predictions
                    FROM predictions
                    WHERE date_heure BETWEEN :start_date AND :end_date
                      AND quartier = :quartier
                    GROUP BY DATE(date_heure)
                    ORDER BY date
                """)
                params = {'start_date': start_date, 'end_date': end_date, 'quartier': quartier}
            else:
                query = text("""
                    SELECT 
                        DATE(date_heure) as date,
                        AVG(risque_global) as risque_moyen,
                        COUNT(*) as nb_predictions
                    FROM predictions
                    WHERE date_heure BETWEEN :start_date AND :end_date
                    GROUP BY DATE(date_heure)
                    ORDER BY date
                """)
                params = {'start_date': start_date, 'end_date': end_date}
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params=params)
            
            return df
            
        except Exception as e:
            st.warning(f"⚠️ Erreur stats période : {e}")
            return pd.DataFrame()
    
    def clear_old_predictions(self, days=30):
        """Supprimer les prédictions de plus de X jours"""
        if not self.engine:
            return 0
        
        try:
            query = text("""
                DELETE FROM predictions
                WHERE date_heure < NOW() - INTERVAL ':days days'
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'days': days})
                conn.commit()
                return result.rowcount
            
        except Exception as e:
            st.warning(f"⚠️ Erreur nettoyage : {e}")
            return 0

# Instance globale
@st.cache_resource
def get_db():
    """Obtenir l'instance de la base de données (avec cache)"""
    return SupabaseDB()