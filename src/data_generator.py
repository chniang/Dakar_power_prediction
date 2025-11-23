# Fichier : src/data_generator.py
# Générateur de données synthétiques pour Dakar Power Prediction
# =================================================================
#
# Ce fichier génère 52,704 enregistrements de données synthétiques (1 an × 6 quartiers).
# Pourquoi synthétiques ? Car je n'ai pas accès aux données réelles de SENELEC.
#
# Les données sont générées de manière RÉALISTE en respectant :
# 1. Les patterns météorologiques de Dakar (température, humidité, vent)
# 2. Les cycles de consommation électrique (pics du soir, creux de nuit)
# 3. Les probabilités de coupure par quartier (Guediawaye > Dakar-Plateau)
# 4. Les corrélations entre variables (chaleur → consommation, surcharge → coupures)

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import de la configuration centrale
try:
    from src.config import (
        START_DATE, END_DATE, QUARTIERS, PROBA_BASE_COUPURE,
        RAW_DATA_FILE, MESSAGES
    )
except ImportError:
    # Si le script est exécuté depuis un autre répertoire, on ajoute le chemin
    sys.path.append(str(Path(__file__).parent.parent))
    from src.config import (
        START_DATE, END_DATE, QUARTIERS, PROBA_BASE_COUPURE,
        RAW_DATA_FILE, MESSAGES
    )


class DataGenerator:
    """
    Générateur de données synthétiques pour la prédiction de coupures électriques.
    
    Cette classe crée un dataset réaliste en simulant :
    - Les conditions météorologiques de Dakar sur 1 an
    - La consommation électrique avec ses variations horaires
    - Les coupures d'électricité selon des probabilités par quartier
    
    Exemple d'utilisation :
        generator = DataGenerator()
        df = generator.generate(save=True)
    """
    
    def __init__(self, start_date=START_DATE, end_date=END_DATE):
        """
        Initialise le générateur avec la période de génération.
        
        Args:
            start_date (str): Date de début au format 'YYYY-MM-DD' (défaut: '2024-01-01')
            end_date (str): Date de fin au format 'YYYY-MM-DD' (défaut: '2025-01-01')
        
        Note:
            La plage de dates génère des enregistrements HORAIRES.
            1 an = 8,784 heures × 6 quartiers = 52,704 enregistrements
        """
        self.start_date = start_date
        self.end_date = end_date
        self.quartiers = QUARTIERS  # Les 6 quartiers de Dakar
        self.proba_base = PROBA_BASE_COUPURE  # Probabilités de base par quartier
        
        # Génération de la plage de dates (horaire)
        # freq='h' → 1 enregistrement par heure
        # inclusive='left' → Exclut la dernière date (end_date)
        self.date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq='h',
            inclusive='left'
        )
        self.n_records = len(self.date_range)
        
        # Affichage des infos de génération
        print(f"📅 Période : {start_date} → {end_date}")
        print(f"📊 Nombre d'heures : {self.n_records:,}")
        print(f"🏘️ Nombre de quartiers : {len(self.quartiers)}")
    
    def generate_weather_features(self, df):
        """
        Génère les variables météorologiques réalistes pour Dakar.
        
        Args:
            df (pd.DataFrame): DataFrame contenant la colonne 'date_heure'
            
        Returns:
            pd.DataFrame: DataFrame enrichi avec les colonnes météo
        
        Variables générées :
            - temp_celsius : Température en °C (15-40°C)
            - humidite_percent : Humidité relative en % (40-95%)
            - vitesse_vent : Vitesse du vent en km/h (0-50 km/h)
        
        Logique :
            La température et l'humidité suivent des cycles saisonniers (sinus).
            Dakar a 2 saisons : chaude (avril-octobre) et fraîche (novembre-mars).
        """
        n = len(df)
        
        # --- Température ---
        # Variation saisonnière : Plus chaud vers juillet (jour 196), plus frais en janvier
        # day_of_year va de 1 à 365
        day_of_year = df['date_heure'].dt.dayofyear
        
        # Base : 25°C (température moyenne annuelle de Dakar)
        # Amplitude : ±5°C de variation saisonnière
        # sin(2π × jour/365) crée un cycle qui se répète chaque année
        temp_base = 25 + 5 * np.sin(2 * np.pi * (day_of_year / 365))
        
        # Ajout de bruit aléatoire (variabilité quotidienne)
        # normal(0, 2) → Moyenne 0, écart-type 2°C
        df['temp_celsius'] = np.round(
            temp_base + np.random.normal(0, 2, n),
            1  # Arrondi à 1 décimale
        )
        
        # --- Humidité ---
        # Plus élevée pendant la saison des pluies (juin-octobre)
        # Inverse de la température (quand il fait chaud, l'air est plus sec)
        hum_base = 65 + 10 * np.sin(2 * np.pi * (day_of_year / 365))
        
        # Ajout de bruit + limitation entre 40% et 95%
        df['humidite_percent'] = np.clip(
            np.round(hum_base + np.random.normal(0, 5, n), 1),
            40,  # Minimum (saison sèche)
            95   # Maximum (saison des pluies)
        )
        
        # --- Vitesse du Vent ---
        # Distribution normale : moyenne 15 km/h, écart-type 8 km/h
        # Les alizés maritimes à Dakar soufflent régulièrement
        # Limitation entre 0 et 50 km/h (vents extrêmes rares)
        df['vitesse_vent'] = np.clip(
            np.round(np.random.normal(15, 8, n), 1),
            0,   # Pas de vent négatif !
            50   # Vent très fort (rare)
        )
        
        return df
    
    def generate_consumption(self, df):
        """
        Génère la consommation électrique avec profil horaire réaliste.
        
        Args:
            df (pd.DataFrame): DataFrame avec colonnes météo
            
        Returns:
            pd.DataFrame: DataFrame avec colonne 'conso_megawatt' ajoutée
        
        Logique de consommation :
            - SOIRÉE/NUIT (18h-6h) : Forte consommation (~800 MW)
              → Éclairage, climatisation, télévision, cuisine
            - JOURNÉE (7h-17h) : Consommation moyenne (~500 MW)
              → Activité commerciale, moins de résidentiel
            - EFFET TEMPÉRATURE : Si > 28°C → +climatisation
        
        Exemple :
            22h, 35°C → 800 MW (base) + 350 MW (clim) = 1150 MW
        """
        n = len(df)
        hour = df['date_heure'].dt.hour
        
        # --- Profil de consommation horaire ---
        # Utilisation de np.where pour condition vectorisée (rapide)
        # Condition : (18h ≤ heure) OU (heure ≤ 6h)
        conso_base = np.where(
            (hour >= 18) | (hour <= 6),
            np.random.normal(800, 100, n),  # Soirée/nuit : 800 MW ± 100 MW
            np.random.normal(500, 80, n)    # Journée : 500 MW ± 80 MW
        )
        
        # --- Effet de la température (climatisation) ---
        # Si temp > 28°C → Chaque degré supplémentaire ajoute 50 MW
        # Exemple : 35°C → (35-28) × 50 = 350 MW supplémentaires
        # clip(x - 28, 0, None) → Si x < 28, retourne 0 (pas d'effet négatif)
        temp_effect = np.clip(df['temp_celsius'] - 28, 0, None) * 50
        
        # --- Consommation finale ---
        # Base + effet température + bruit aléatoire
        df['conso_megawatt'] = np.round(
            conso_base + temp_effect + np.random.normal(0, 50, n),
            1
        )
        
        # Limitation réaliste : 400 MW (creux) à 1200 MW (pic absolu)
        df['conso_megawatt'] = np.clip(df['conso_megawatt'], 400, 1200)
        
        return df
    
    def generate_outages(self, df):
        """
        Génère les coupures d'électricité (variable cible à prédire).
        
        Args:
            df (pd.DataFrame): DataFrame avec toutes les features
            
        Returns:
            pd.DataFrame: DataFrame avec colonne 'coupure' (0 ou 1)
        
        Logique des coupures :
            La probabilité de coupure dépend de 4 facteurs :
            
            1. QUARTIER (facteur principal) :
               - Guediawaye : 12% de base (infrastructure fragile)
               - Dakar-Plateau : 2% de base (infrastructure moderne)
            
            2. CONSOMMATION :
               - Si consommation > 800 MW → Risque de surcharge réseau
               - Exemple : 1200 MW → +20% de risque
            
            3. TEMPÉRATURE :
               - Si température > 35°C → Câbles surchauffent
               - Exemple : 38°C → +20% de risque
            
            4. VENT FORT :
               - Si vent > 40 km/h → Lignes endommagées
               - Exemple : 45 km/h → +8% de risque
        
        Note importante :
            Les facteurs de multiplication (×4) sont utilisés pour amplifier
            les effets et obtenir un taux de coupure réaliste (~7% global).
        """
        n = len(df)
        
        # --- 1. Probabilité de base par quartier ---
        proba_coupure = np.zeros(n)
        for quartier, base_prob in self.proba_base.items():
            # Masque booléen : True pour toutes les lignes de ce quartier
            mask = df['quartier'] == quartier
            proba_coupure[mask] += base_prob
        
        # --- 2. Influence de la CONSOMMATION (risque de surcharge) ---
        # Formule : (consommation - 800) / 400 donne un ratio 0-1
        # Si conso = 800 MW → ratio = 0 (pas de risque)
        # Si conso = 1200 MW → ratio = 1 (risque maximal)
        # × 0.05 × 4 = jusqu'à +20% de probabilité
        conso_risk = np.clip((df['conso_megawatt'] - 800) / 400, 0, 1) * 0.05 * 4
        proba_coupure += conso_risk
        
        # --- 3. Influence de la TEMPÉRATURE extrême ---
        # Si température > 35°C → +20% de probabilité
        # Raison : Les câbles électriques surchauffent et peuvent fondre
        temp_risk = np.where(df['temp_celsius'] > 35, 0.05 * 4, 0)
        proba_coupure += temp_risk
        
        # --- 4. Influence du VENT fort ---
        # Si vent > 40 km/h → +8% de probabilité
        # Raison : Branches d'arbres qui tombent sur les lignes
        wind_risk = np.where(df['vitesse_vent'] > 40, 0.02 * 4, 0)
        proba_coupure += wind_risk
        
        # --- Limitation de la probabilité finale ---
        # Maximum 50% (même dans les pires conditions, pas 100% de coupures)
        proba_coupure = np.clip(proba_coupure, 0, 0.5)
        
        # --- Génération binaire des coupures ---
        # binomial(1, p) → Tire 0 ou 1 selon la probabilité p
        # Exemple : Si p=0.15 → 15% de chance d'avoir 1 (coupure)
        df['coupure'] = np.random.binomial(1, proba_coupure)
        
        return df
    
    def generate(self, save=True):
        """
        Génère le dataset complet en orchestrant toutes les étapes.
        
        Args:
            save (bool): Si True, sauvegarde dans data/raw/raw_data.csv
            
        Returns:
            pd.DataFrame: Dataset complet avec 52,704 lignes × 8 colonnes
        
        Pipeline de génération :
            1. Créer le DataFrame de base (dates horaires)
            2. Générer les features météo (température, humidité, vent)
            3. Générer la consommation électrique
            4. Répliquer pour les 6 quartiers
            5. Générer les coupures (variable cible)
            6. Ajouter les identifiants
            7. Afficher les statistiques
            8. Sauvegarder en CSV
        """
        print("\n" + "="*50)
        print("🔄 GÉNÉRATION DES DONNÉES SYNTHÉTIQUES")
        print("="*50)
        
        # --- Étape 1 : DataFrame de base ---
        print("\n1️⃣ Création du DataFrame de base...")
        df = pd.DataFrame(self.date_range, columns=['date_heure'])
        
        # Ajout de colonnes temporelles (pour features engineering ultérieur)
        df['heure'] = df['date_heure'].dt.hour          # 0-23
        df['jour_semaine'] = df['date_heure'].dt.dayofweek  # 0=Lundi, 6=Dimanche
        
        # --- Étape 2 : Météo ---
        print("2️⃣ Génération des variables météorologiques...")
        df = self.generate_weather_features(df)
        
        # --- Étape 3 : Consommation ---
        print("3️⃣ Génération de la consommation électrique...")
        df = self.generate_consumption(df)
        
        # --- Étape 4 : Répliquer pour chaque quartier ---
        print("4️⃣ Réplication pour chaque quartier...")
        # assign(quartier=q) crée une copie du df avec la colonne 'quartier'
        # concat(...) empile verticalement les 6 DataFrames
        # Résultat : 8,784 heures × 6 quartiers = 52,704 lignes
        df_all = pd.concat(
            [df.assign(quartier=q) for q in self.quartiers],
            ignore_index=True  # Recréer un index 0, 1, 2, ...
        )
        
        # --- Étape 5 : Coupures ---
        print("5️⃣ Génération des coupures (variable cible)...")
        df_all = self.generate_outages(df_all)
        
        # --- Étape 6 : Identifiants ---
        print("6️⃣ Ajout des identifiants...")
        # Insert en position 0 (première colonne)
        df_all.insert(0, 'id_enregistrement', range(1, len(df_all) + 1))
        
        # --- Étape 7 : Réorganiser les colonnes ---
        # Ordre logique : ID, date, localisation, météo, consommation, cible
        df_all = df_all[[
            'id_enregistrement', 'date_heure', 'quartier',
            'temp_celsius', 'humidite_percent', 'vitesse_vent',
            'conso_megawatt', 'coupure'
        ]]
        
        # --- Étape 8 : Statistiques ---
        self._print_statistics(df_all)
        
        # --- Étape 9 : Sauvegarde ---
        if save:
            self._save_data(df_all)
        
        print("\n" + "="*50)
        print(MESSAGES['data_generated'])
        print("="*50 + "\n")
        
        return df_all
    
    def _print_statistics(self, df):
        """
        Affiche des statistiques détaillées sur le dataset généré.
        
        Cela permet de vérifier rapidement que :
        - Les données sont réalistes (pas de valeurs aberrantes)
        - Les taux de coupure sont cohérents par quartier
        - Les distributions météo sont correctes
        """
        print("\n" + "="*50)
        print("📊 STATISTIQUES DU DATASET GÉNÉRÉ")
        print("="*50)
        
        # --- Taille totale ---
        print(f"\n📏 Taille totale : {len(df):,} enregistrements")
        print(f"📅 Période : {df['date_heure'].min()} → {df['date_heure'].max()}")
        
        # --- Statistiques sur les coupures ---
        print("\n🔌 Statistiques sur les coupures :")
        coupure_stats = df['coupure'].value_counts(normalize=True).mul(100).round(2)
        print(f"  • Pas de coupure : {coupure_stats.get(0, 0):.2f}%")
        print(f"  • Coupure        : {coupure_stats.get(1, 0):.2f}%")
        print(f"  • Total coupures : {df['coupure'].sum():,}")
        
        # --- Taux par quartier (du plus risqué au moins risqué) ---
        print("\n🏘️ Taux de coupures par quartier :")
        quartier_stats = df.groupby('quartier')['coupure'].agg(['mean', 'sum'])
        quartier_stats['mean'] = quartier_stats['mean'].mul(100).round(2)
        quartier_stats = quartier_stats.sort_values('mean', ascending=False)
        
        for quartier, row in quartier_stats.iterrows():
            # Formatage avec espaces pour alignement
            print(f"  • {quartier:25} : {row['mean']:5.2f}% ({int(row['sum'])} coupures)")
        
        # --- Statistiques météo ---
        print("\n🌡️ Statistiques météorologiques :")
        print(f"  • Température : {df['temp_celsius'].min():.1f}°C → {df['temp_celsius'].max():.1f}°C (moy: {df['temp_celsius'].mean():.1f}°C)")
        print(f"  • Humidité    : {df['humidite_percent'].min():.1f}% → {df['humidite_percent'].max():.1f}% (moy: {df['humidite_percent'].mean():.1f}%)")
        print(f"  • Vent        : {df['vitesse_vent'].min():.1f} → {df['vitesse_vent'].max():.1f} km/h (moy: {df['vitesse_vent'].mean():.1f} km/h)")
        
        # --- Statistiques consommation ---
        print("\n⚡ Statistiques de consommation électrique :")
        print(f"  • Min  : {df['conso_megawatt'].min():.1f} MW")
        print(f"  • Max  : {df['conso_megawatt'].max():.1f} MW")
        print(f"  • Moy  : {df['conso_megawatt'].mean():.1f} MW")
        print(f"  • Med  : {df['conso_megawatt'].median():.1f} MW")
        
        # --- Aperçu des 10 premières lignes ---
        print("\n📋 Aperçu des 10 premières lignes :")
        print(df.head(10).to_string(index=False))
    
    def _save_data(self, df):
        """
        Sauvegarde le dataset dans data/raw/raw_data.csv
        
        Args:
            df (pd.DataFrame): Dataset à sauvegarder
        """
        # Créer le dossier parent si nécessaire
        RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder en CSV (index=False pour ne pas sauvegarder l'index pandas)
        df.to_csv(RAW_DATA_FILE, index=False)
        
        print(f"\n💾 Données sauvegardées : {RAW_DATA_FILE}")
        # Afficher la taille du fichier en MB
        file_size_mb = RAW_DATA_FILE.stat().st_size / 1024 / 1024
        print(f"   Taille du fichier : {file_size_mb:.2f} MB")


def main():
    """
    Fonction principale pour générer les données.
    
    Cette fonction est appelée quand on exécute :
        python src/data_generator.py
    
    Elle crée le générateur et génère les 52,704 enregistrements.
    """
    # Créer une instance du générateur
    generator = DataGenerator()
    
    # Générer et sauvegarder les données
    df = generator.generate(save=True)
    
    return df


# Point d'entrée du script
if __name__ == "__main__":
    main()