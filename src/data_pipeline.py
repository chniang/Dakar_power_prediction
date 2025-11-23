# Fichier : src/data_pipeline.py
# Pipeline de prétraitement des données avec SPLIT CHRONOLOGIQUE SIMPLE
# =======================================================================
#
# Ce fichier transforme les données brutes (raw_data.csv) en données prêtes
# pour le Machine Learning. C'est l'étape CRUCIALE entre les données et le modèle.
#
# Pourquoi un pipeline ? Pour garantir que les mêmes transformations sont appliquées
# pendant l'entraînement ET pendant la prédiction (cohérence).
#
# Étapes du pipeline :
# 1. Charger les données brutes
# 2. Tri chronologique (IMPORTANT : pas par quartier !)
# 3. Feature engineering (créer heure, jour, mois, is_peak_hour)
# 4. Encodage des quartiers (texte → nombres)
# 5. Split train/test chronologique (80/20)
# 6. Normalisation (StandardScaler sur train, puis appliqué sur test)
#
# ⚠️ CORRECTION V6 : Suppression de la stratification par quartier qui cassait
# l'ordre temporel et causait l'inversion des prédictions.

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys

# Import de la configuration
try:
    from src.config import (
        RAW_DATA_FILE, PROCESSED_DATA_FILE,
        SCALER_FILE, ENCODERS_FILE,
        FEATURE_COLUMNS, FEATURES_TO_SCALE, TARGET_COLUMN,
        TEST_SIZE, RANDOM_STATE, SEQUENCE_LENGTH,
        MESSAGES
    )
except ImportError:
    # Si exécuté depuis un autre répertoire
    sys.path.append(str(Path(__file__).parent.parent))
    from src.config import (
        RAW_DATA_FILE, PROCESSED_DATA_FILE,
        SCALER_FILE, ENCODERS_FILE,
        FEATURE_COLUMNS, FEATURES_TO_SCALE, TARGET_COLUMN,
        TEST_SIZE, RANDOM_STATE, SEQUENCE_LENGTH,
        MESSAGES
    )


class DataPipeline:
    """
    Pipeline de prétraitement des données pour Dakar Power Prediction.
    
    Cette classe orchestre toutes les transformations nécessaires pour passer
    des données brutes aux features normalisées utilisables par les modèles ML.
    
    Responsabilités :
    - Chargement des données
    - Feature engineering (création de nouvelles variables)
    - Encodage des variables catégorielles (quartier → nombre)
    - Normalisation (StandardScaler)
    - Split train/test chronologique
    - Création de séquences pour LSTM
    
    Exemple d'utilisation :
        pipeline = DataPipeline()
        data = pipeline.process_for_training(save_processed=True)
        X_train, y_train = data['X_train'], data['y_train']
    """
    
    def __init__(self):
        """
        Initialise le pipeline avec les transformers vides.
        
        Les transformers (scaler, label_encoder) seront créés lors du fit
        sur les données d'entraînement, puis réutilisés pour le test et la prédiction.
        """
        self.scaler = None              # StandardScaler (normalisation)
        self.label_encoder = None       # LabelEncoder (quartier → 0-5)
        self.feature_columns = FEATURE_COLUMNS  # Les 9 features du modèle
        self.features_to_scale = FEATURES_TO_SCALE  # Colonnes à normaliser
        self.target_column = TARGET_COLUMN  # 'coupure' (0 ou 1)
        
    def load_raw_data(self, file_path=None):
        """
        Charge les données brutes depuis le CSV.
        
        Args:
            file_path (Path): Chemin vers raw_data.csv (défaut: config.RAW_DATA_FILE)
            
        Returns:
            pd.DataFrame: Données brutes avec 52,704 lignes × 8 colonnes
        
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
        
        Note:
            parse_dates=['date_heure'] convertit automatiquement la colonne
            'date_heure' en type datetime (au lieu de string).
        """
        if file_path is None:
            file_path = RAW_DATA_FILE
            
        print(f"📂 Chargement des données : {file_path}")
        
        # Vérifier l'existence du fichier
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {file_path}")
        
        # Charger avec parsing automatique des dates
        df = pd.read_csv(file_path, parse_dates=['date_heure'])
        print(f"   ✅ {len(df):,} enregistrements chargés")
        
        return df
    
    def create_time_features(self, df):
        """
        Crée les features temporelles à partir de la date.
        
        Args:
            df (pd.DataFrame): DataFrame contenant 'date_heure'
            
        Returns:
            pd.DataFrame: DataFrame enrichi avec 4 nouvelles colonnes
        
        Features créées :
            - heure : 0-23 (heure de la journée)
            - jour_semaine : 0-6 (0=Lundi, 6=Dimanche)
            - mois : 1-12 (mois de l'année)
            - is_peak_hour : 0/1 (heure de pointe ou non)
        
        Logique is_peak_hour :
            Les heures de pointe sont définies comme :
            - Soirée/nuit : 18h-6h (forte consommation résidentielle)
            - Matin : 6h-8h (pic matinal)
            
            Pourquoi ? C'est pendant ces heures que le réseau est le plus sollicité,
            donc le risque de coupure est plus élevé.
        """
        print("🕐 Création des features temporelles...")
        
        df = df.copy()  # Éviter de modifier le DataFrame original
        
        # Extraction des composantes temporelles
        # dt.hour, dt.dayofweek, dt.month sont des accesseurs pandas pour datetime
        df['heure'] = df['date_heure'].dt.hour
        df['jour_semaine'] = df['date_heure'].dt.dayofweek  # 0=Lundi, 6=Dimanche
        df['mois'] = df['date_heure'].dt.month
        
        # Feature binaire : heure de pointe
        # Logique : (18h ≤ heure ≤ 23h) OU (0h ≤ heure ≤ 6h) OU (6h ≤ heure ≤ 8h)
        # Opérateur | = OU logique (vectorisé sur tout le DataFrame)
        df['is_peak_hour'] = (
            ((df['heure'] >= 18) | (df['heure'] <= 6)) |  # Soirée/nuit
            (df['heure'].between(6, 8, inclusive='both'))   # Matin
        ).astype(int)  # Convertir bool → int (True=1, False=0)
        
        print(f"   ✅ Features temporelles créées")
        
        return df
    
    def encode_categorical(self, df, fit=True):
        """
        Encode les variables catégorielles (quartier) en nombres.
        
        Args:
            df (pd.DataFrame): DataFrame avec colonne 'quartier'
            fit (bool): Si True, crée et fit l'encodeur. Si False, utilise l'existant
            
        Returns:
            pd.DataFrame: DataFrame avec colonne 'quartier_encoded' ajoutée
        
        Encodage :
            LabelEncoder transforme les noms de quartiers en nombres :
            'Dakar-Plateau' → 0
            'Guediawaye' → 1
            'Mermoz-Sacré-Coeur' → 2
            etc.
        
        Pourquoi encoder ?
            Les algorithmes ML ne peuvent pas traiter directement du texte.
            Il faut convertir en nombres.
        
        ⚠️ Important : 
            - En mode training (fit=True) : On crée l'encodeur et on le sauvegarde
            - En mode prediction (fit=False) : On charge l'encodeur existant
            
            Pourquoi ? Pour garantir que 'Guediawaye' sera toujours encodé en 1,
            même dans de nouvelles données.
        """
        print("🏷️ Encodage des variables catégorielles...")
        
        df = df.copy()
        
        if fit:
            # MODE TRAINING : Créer et fitter l'encodeur
            self.label_encoder = LabelEncoder()
            df['quartier_encoded'] = self.label_encoder.fit_transform(df['quartier'])
            
            # Sauvegarder pour réutilisation ultérieure
            ENCODERS_FILE.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({'quartier': self.label_encoder}, ENCODERS_FILE)
            print(f"   💾 Encodeur sauvegardé : {ENCODERS_FILE}")
            
        else:
            # MODE PREDICTION : Charger l'encodeur existant
            if not ENCODERS_FILE.exists():
                raise FileNotFoundError(f"Encodeur non trouvé : {ENCODERS_FILE}")
            
            encoders = joblib.load(ENCODERS_FILE)
            self.label_encoder = encoders['quartier']
            
            # Encoder avec gestion des valeurs inconnues
            # Si un quartier n'était pas dans le training, on met -1
            df['quartier_encoded'] = df['quartier'].apply(
                lambda x: self.label_encoder.transform([x])[0] 
                if x in self.label_encoder.classes_ 
                else -1  # Valeur par défaut pour quartier inconnu
            )
            print(f"   📂 Encodeur chargé : {ENCODERS_FILE}")
        
        print(f"   ✅ Encodage terminé")
        
        return df
    
    def scale_features(self, df, fit=True):
        """
        Normalise les features numériques avec StandardScaler.
        
        Args:
            df (pd.DataFrame): DataFrame avec features numériques
            fit (bool): Si True, fit le scaler. Si False, utilise le scaler existant
            
        Returns:
            pd.DataFrame: DataFrame avec features normalisées
        
        StandardScaler :
            Formule : X_scaled = (X - mean) / std
            
            Exemple :
            temp_celsius = [20, 25, 30]
            mean = 25°C, std = 5°C
            
            20°C → (20-25)/5 = -1.0
            25°C → (25-25)/5 =  0.0
            30°C → (30-25)/5 = +1.0
        
        Pourquoi normaliser ?
            1. Éviter que certaines features dominent (ex: conso_megawatt en MW
               vs temp_celsius en °C → échelles très différentes)
            2. Accélérer la convergence des algorithmes ML
            3. OBLIGATOIRE pour LSTM (stabilité de l'entraînement)
        
        Features normalisées (config.FEATURES_TO_SCALE) :
            - temp_celsius : 15-40°C
            - vitesse_vent : 0-50 km/h
            - conso_megawatt : 400-1200 MW
        
        Features NON normalisées :
            - heure (0-23), jour_semaine (0-6), mois (1-12) → Pas besoin
            - humidite_percent → Retiré car causait des problèmes de corrélation
        """
        print("📊 Normalisation des features...")
        
        df = df.copy()
        
        # Sélectionner uniquement les colonnes qui existent dans le DataFrame
        cols_to_scale = [col for col in self.features_to_scale if col in df.columns]
        
        if fit:
            # MODE TRAINING : Créer et fitter le scaler
            self.scaler = StandardScaler()
            # fit_transform() calcule mean et std, puis normalise
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            
            # Sauvegarder le scaler (avec mean et std mémorisés)
            SCALER_FILE.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, SCALER_FILE)
            print(f"   💾 Scaler sauvegardé : {SCALER_FILE}")
            
        else:
            # MODE PREDICTION : Charger le scaler existant
            if not SCALER_FILE.exists():
                raise FileNotFoundError(f"Scaler non trouvé : {SCALER_FILE}")
            
            self.scaler = joblib.load(SCALER_FILE)
            # transform() utilise les mean et std du training (pas de fit !)
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
            print(f"   📂 Scaler chargé : {SCALER_FILE}")
        
        print(f"   ✅ Normalisation terminée")
        
        return df
    
    def prepare_features(self, df, include_target=True):
        """
        Prépare les features finales pour le modèle.
        
        Args:
            df (pd.DataFrame): DataFrame prétraité
            include_target (bool): Si True, retourne (X, y). Si False, retourne X uniquement
            
        Returns:
            tuple: (X, y) si include_target=True, sinon X uniquement
        
        Cette fonction sélectionne uniquement les 9 colonnes nécessaires au modèle
        (définies dans config.FEATURE_COLUMNS) et sépare X (features) de y (target).
        
        Ordre des features (IMPORTANT) :
            1. temp_celsius
            2. humidite_percent
            3. vitesse_vent
            4. conso_megawatt
            5. heure
            6. jour_semaine
            7. mois
            8. is_peak_hour
            9. quartier_encoded
        """
        print("🎯 Préparation des features finales...")
        
        # Sélectionner uniquement les features qui existent
        available_features = [col for col in self.feature_columns if col in df.columns]
        X = df[available_features].copy()
        
        print(f"   ✅ {len(available_features)} features sélectionnées")
        
        if include_target:
            # Mode training : Retourner X et y
            if self.target_column not in df.columns:
                raise ValueError(f"Colonne cible '{self.target_column}' non trouvée")
            
            y = df[self.target_column].copy()
            return X, y
        
        # Mode prediction : Retourner uniquement X
        return X
    
    def create_sequences(self, X, y=None, sequence_length=SEQUENCE_LENGTH):
        """
        Crée des séquences temporelles pour le LSTM.
        
        Args:
            X (np.ndarray): Features (n_samples, n_features)
            y (np.ndarray): Target (optionnel)
            sequence_length (int): Longueur des séquences (défaut: 12 heures)
            
        Returns:
            tuple: (X_seq, y_seq) si y fourni, sinon X_seq uniquement
        
        Principe :
            Le LSTM a besoin d'historique pour prédire. On crée des fenêtres
            glissantes de 12 heures.
        
        Exemple avec sequence_length=3 :
            X = [[x1], [x2], [x3], [x4], [x5]]
            y = [y1, y2, y3, y4, y5]
            
            Séquences créées :
            X_seq[0] = [x1, x2, x3]  →  y_seq[0] = y4 (prédire heure 4 avec 1-2-3)
            X_seq[1] = [x2, x3, x4]  →  y_seq[1] = y5 (prédire heure 5 avec 2-3-4)
        
        Shape finale :
            X : (n_samples, n_features) = (52704, 9)
            X_seq : (n_samples - 12, 12, 9) = (52692, 12, 9)
            
            Explication : On perd 12 échantillons car on ne peut pas créer
            de séquence pour les 12 premières heures (pas d'historique).
        
        ⚠️ Correction V6 :
            Conversion en numpy array AVANT la boucle pour éviter les problèmes
            d'indexation avec pandas Series.
        """
        print(f"🔄 Création des séquences (longueur={sequence_length})...")
        
        # Convertir en numpy array pour éviter les problèmes d'index pandas
        # Si X est déjà un ndarray, on ne fait rien
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        if y is not None:
            y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        X_seq = []
        y_seq = []
        
        # Boucle sur les échantillons (à partir de l'index sequence_length)
        for i in range(sequence_length, len(X)):
            # Créer une séquence : 12 heures précédentes
            X_seq.append(X[i-sequence_length:i])
            
            # Target : heure actuelle
            if y is not None:
                y_seq.append(y[i])
        
        # Convertir les listes en arrays numpy
        X_seq = np.array(X_seq)
        
        print(f"   ✅ {len(X_seq):,} séquences créées")
        
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        
        return X_seq
    
    def split_data_chronological_stratified(self, X, y, test_size=TEST_SIZE):
        """
        ✅ CORRECTION V6 : Split chronologique SIMPLE (80/20)
        
        Pourquoi "chronologique simple" ?
            Avant, j'avais un split stratifié par quartier qui cassait l'ordre
            temporel et causait l'inversion des prédictions.
        
        Maintenant :
            - On trie les données par date UNIQUEMENT (pas par quartier)
            - On prend les 80% premiers pour train
            - On prend les 20% derniers pour test
        
        Args:
            X: Features (DataFrame ou ndarray)
            y: Target (Series ou ndarray)
            test_size (float): Proportion du test set (défaut: 0.2)
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        
        Exemple :
            52,704 enregistrements
            Split à l'index 42,163 (80%)
            
            Train : indices 0 à 42,162 (Janvier-Octobre 2024)
            Test : indices 42,163 à 52,703 (Novembre-Décembre 2024)
        
        ⚠️ CRITIQUE : Pourquoi pas de split aléatoire ?
            Split aléatoire → FUITE DE DONNÉES !
            Le modèle verrait des données futures pendant le training.
            
            Exemple :
            Train : [Janvier, Mars, Mai, Juillet]
            Test : [Février, Avril, Juin, Août]
            → Le modèle a vu Mars pendant training, puis doit prédire Février (le passé) !
        
        ⚠️ Pourquoi pas de stratification par quartier ?
            Stratifier par quartier → CASSE L'ORDRE TEMPOREL !
            On mélangerait les dates pour garantir 80/20 par quartier.
            → Causerait l'inversion des prédictions (problème résolu en V6).
        """
        print(f"✂️ Séparation CHRONOLOGIQUE SIMPLE ({int((1-test_size)*100)}%/{int(test_size*100)})...")
        
        # Calculer l'index de séparation
        # int() arrondit à l'entier inférieur
        split_idx = int(len(X) * (1 - test_size))
        
        # Split simple selon le type (DataFrame ou array)
        if isinstance(X, pd.DataFrame):
            # iloc = sélection par position (0 to split_idx)
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
        else:
            # Slicing numpy
            X_train = X[:split_idx]
            X_test = X[split_idx:]
        
        # Target (fonctionne pour Series et array)
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # Affichage des statistiques
        print(f"   ✅ Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   ✅ Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
        print(f"   📅 Split chronologique simple (pas de stratification)")
        
        # Afficher les taux de coupure (vérifier qu'ils sont similaires)
        # Si trop différents → Problème de représentativité
        train_rate = y_train.mean() * 100
        test_rate = y_test.mean() * 100
        print(f"   📊 Train - Taux coupure: {train_rate:.2f}%")
        print(f"   📊 Test  - Taux coupure: {test_rate:.2f}%")
        
        return X_train, X_test, y_train, y_test
    
    def process_for_training(self, save_processed=True):
        """
        Pipeline complet pour l'entraînement des modèles.
        
        Args:
            save_processed (bool): Si True, sauvegarde processed_data.csv
            
        Returns:
            dict: Dictionnaire contenant toutes les données préparées
            {
                'X_train': ndarray normalisé,
                'X_test': ndarray normalisé,
                'y_train': ndarray,
                'y_test': ndarray,
                'feature_names': liste des noms de features,
                'scaler': StandardScaler fitté,
                'label_encoder': LabelEncoder fitté
            }
        
        Ce dictionnaire contient tout ce qui est nécessaire pour :
        1. Entraîner les modèles (X_train, y_train)
        2. Évaluer les modèles (X_test, y_test)
        3. Faire des prédictions futures (scaler, label_encoder)
        
        Pipeline en 7 étapes :
            1. Charger raw_data.csv
            2. ✅ Trier par date UNIQUEMENT (correction V6)
            3. Feature engineering (heure, jour, mois, is_peak_hour)
            4. Encoder quartiers (texte → nombres)
            5. Split 80/20 chronologique
            6. Normaliser (fit sur train, transform sur test)
            7. Retourner tout dans un dict
        """
        print("\n" + "="*60)
        print("🔄 PIPELINE DE PRÉTRAITEMENT V6 - MODE TRAINING")
        print("="*60 + "\n")
        
        # --- Étape 1 : Chargement ---
        df = self.load_raw_data()
        
        # --- Étape 2 : ✅ CORRECTION V6 - Tri chronologique simple ---
        # AVANT (V5 et antérieurs) : df.sort_values(['quartier', 'date_heure'])
        # → Triait par quartier d'abord, ce qui cassait l'ordre temporel global
        # → Causait l'inversion des prédictions
        #
        # APRÈS (V6) : df.sort_values('date_heure')
        # → Trie uniquement par date, ordre chronologique pur
        print("📅 Tri chronologique (par date uniquement)...")
        df = df.sort_values('date_heure').reset_index(drop=True)
        print(f"   ✅ Données triées de {df['date_heure'].min()} à {df['date_heure'].max()}")
        
        # --- Étape 3 : Feature engineering ---
        df = self.create_time_features(df)
        
        # --- Étape 4 : Encodage ---
        df = self.encode_categorical(df, fit=True)
        
        # --- Étape 5 : Sauvegarde intermédiaire (avant normalisation) ---
        # Utile pour l'analyse exploratoire des données (EDA)
        if save_processed:
            PROCESSED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(PROCESSED_DATA_FILE, index=False)
            print(f"💾 Données prétraitées sauvegardées : {PROCESSED_DATA_FILE}\n")
        
        # --- Étape 6 : Préparation X et y ---
        X, y = self.prepare_features(df, include_target=True)
        
        # --- Étape 7 : Split train/test ---
        X_train, X_test, y_train, y_test = self.split_data_chronological_stratified(X, y)
        
        # --- Étape 8 : Normalisation ---
        # CRITIQUE : fit sur train UNIQUEMENT !
        # Si on fit sur train+test → FUITE DE DONNÉES (le modèle verrait le futur)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)  # fit + transform
        X_test_scaled = self.scaler.transform(X_test)        # transform uniquement
        
        # Sauvegarder le scaler pour réutilisation
        joblib.dump(self.scaler, SCALER_FILE)
        print(f"💾 Scaler sauvegardé : {SCALER_FILE}\n")
        
        print("="*60)
        print("✅ PRÉTRAITEMENT TERMINÉ (V6)")
        print("="*60 + "\n")
        
        # Retourner tout dans un dictionnaire pratique
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X_train.columns.tolist() if hasattr(X_train, 'columns') else self.feature_columns,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
    
    def process_for_prediction(self, df):
        """
        Pipeline pour la prédiction sur de nouvelles données.
        
        Args:
            df (pd.DataFrame): Nouvelles données (sans colonne 'coupure')
            
        Returns:
            np.ndarray: Features normalisées prêtes pour model.predict()
        
        Différences avec process_for_training :
            - Pas de fit (utilise les transformers sauvegardés)
            - Pas de split train/test
            - Pas de colonne cible
        
        Utilisation dans Streamlit :
            Quand l'utilisateur bouge les sliders (temp, humidité, etc.),
            on crée un DataFrame avec ces valeurs, on applique ce pipeline,
            puis on fait model.predict(X_scaled).
        """
        print("🔮 Prétraitement pour prédiction...")
        
        # 1. Features temporelles
        df = self.create_time_features(df)
        
        # 2. Encoder (utilise l'encodeur existant, fit=False)
        df = self.encode_categorical(df, fit=False)
        
        # 3. Préparer X (sans y car on ne connaît pas encore la coupure)
        X = self.prepare_features(df, include_target=False)
        
        # 4. Normaliser (utilise le scaler existant)
        if self.scaler is None:
            # Si pas encore chargé, charger depuis le fichier
            self.scaler = joblib.load(SCALER_FILE)
        
        X_scaled = self.scaler.transform(X)
        
        print(f"   ✅ {len(X_scaled)} échantillons prêts pour prédiction")
        
        return X_scaled


def main():
    """
    Fonction de test du pipeline.
    
    Exécutée quand on lance : python src/data_pipeline.py
    
    Permet de vérifier que le pipeline fonctionne correctement
    et d'afficher les shapes des données générées.
    """
    pipeline = DataPipeline()
    
    # Tester le pipeline complet
    data = pipeline.process_for_training(save_processed=True)
    
    # Afficher un résumé
    print("\n📊 Résumé des données préparées :")
    print(f"   • X_train shape : {data['X_train'].shape}")
    print(f"   • X_test shape  : {data['X_test'].shape}")
    print(f"   • y_train shape : {data['y_train'].shape}")
    print(f"   • y_test shape  : {data['y_test'].shape}")
    print(f"   • Features      : {len(data['feature_names'])}")
    print(f"\n   Features : {data['feature_names']}")
    
    return data


if __name__ == "__main__":
    main()