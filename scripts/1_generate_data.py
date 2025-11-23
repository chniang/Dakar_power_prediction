# Fichier : scripts/1_generate_data.py
# Script pour générer les données synthétiques
# ==============================================
#
# OBJECTIF PRINCIPAL :
# Ce script génère des données synthétiques réalistes simulant les coupures
# d'électricité à Dakar sur une période définie (par défaut 1 an).
#
# POURQUOI GÉNÉRER DES DONNÉES SYNTHÉTIQUES ?
# - Pas d'accès aux données réelles de SENELEC (confidentielles)
# - Besoin de données contrôlées pour tester les modèles
# - Permet de simuler des patterns réalistes (saisonnalité, heures de pointe, etc.)
#
# FONCTIONNALITÉS :
# 1. Génère un dataset CSV avec ~52,000 lignes (1 an, horaire, 6 quartiers)
# 2. Sauvegarde dans data/raw/power_outages.csv
# 3. Optionnel : Importe dans une base de données SQLite
# 4. Affiche des statistiques détaillées
#
# DURÉE : ~5 secondes (génération) + ~10 secondes (import DB si demandé)
#
# UTILISATION :
# python scripts/1_generate_data.py                    # Génération standard
# python scripts/1_generate_data.py --start 2024-01-01 # Période personnalisée
# python scripts/1_generate_data.py --import-db        # Génération + import DB
# python scripts/1_generate_data.py --no-save          # Voir les stats sans sauvegarder

import sys
from pathlib import Path

# === CONFIGURATION DES CHEMINS ===
# Ajouter le dossier parent (racine du projet) au path Python
# Cela permet d'importer les modules depuis src/
project_root = Path(__file__).parent.parent  # Remonte de scripts/ vers racine/
sys.path.append(str(project_root))

import argparse
from src.data_generator import DataGenerator
from src.database import DatabaseManager, import_csv_to_db
from src.config import RAW_DATA_FILE, START_DATE, END_DATE


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale du script de génération de données.
    
    WORKFLOW COMPLET :
    1. Parser les arguments de ligne de commande
    2. Générer les données synthétiques (DataGenerator)
    3. Sauvegarder le CSV (sauf si --no-save)
    4. Optionnel : Importer dans la base de données SQLite
    5. Afficher les statistiques finales
    
    ARGUMENTS ACCEPTÉS :
    --start      : Date de début (format YYYY-MM-DD)
    --end        : Date de fin (format YYYY-MM-DD)
    --no-save    : Ne pas sauvegarder le CSV (mode test)
    --import-db  : Importer les données dans la base de données
    --drop-db    : Supprimer la BD existante avant import (⚠️ DESTRUCTIF)
    
    EXEMPLES D'UTILISATION :
    
    1. Génération standard (1 an de données) :
       python scripts/1_generate_data.py
    
    2. Période personnalisée (6 mois) :
       python scripts/1_generate_data.py --start 2023-01-01 --end 2023-06-30
    
    3. Génération + import dans la BD :
       python scripts/1_generate_data.py --import-db
    
    4. Régénérer complètement la BD (⚠️ efface tout) :
       python scripts/1_generate_data.py --import-db --drop-db
    
    5. Test sans sauvegarde (voir les stats uniquement) :
       python scripts/1_generate_data.py --no-save
    
    Returns:
        DataFrame : Les données générées (pour tests/débogage)
    """
    
    # === ÉTAPE 1 : PARSER LES ARGUMENTS ===
    parser = argparse.ArgumentParser(
        description="Génère les données synthétiques pour Dakar Power Prediction",
        epilog="""
Exemples:
  %(prog)s                              # Génération standard
  %(prog)s --start 2024-01-01           # Période personnalisée
  %(prog)s --import-db                  # Génération + import BD
  %(prog)s --import-db --drop-db        # Régénération complète
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--start', 
        type=str, 
        default=START_DATE,
        help=f"Date de début au format YYYY-MM-DD (défaut: {START_DATE})"
    )
    
    parser.add_argument(
        '--end', 
        type=str, 
        default=END_DATE,
        help=f"Date de fin au format YYYY-MM-DD (défaut: {END_DATE})"
    )
    
    parser.add_argument(
        '--no-save', 
        action='store_true',
        help="Ne pas sauvegarder le CSV (utile pour tester la génération)"
    )
    
    parser.add_argument(
        '--import-db', 
        action='store_true',
        help="Importer les données dans la base de données SQLite après génération"
    )
    
    parser.add_argument(
        '--drop-db', 
        action='store_true',
        help="⚠️ Supprimer et recréer la BD avant import (DESTRUCTIF!)"
    )
    
    args = parser.parse_args()
    
    # === EN-TÊTE DU SCRIPT ===
    print("\n" + "="*70)
    print("📊 SCRIPT 1 : GÉNÉRATION DES DONNÉES SYNTHÉTIQUES")
    print("="*70)
    
    # === ÉTAPE 2 : GÉNÉRER LES DONNÉES ===
    print(f"\n🔄 Génération des données de {args.start} à {args.end}...")
    print(f"   Configuration :")
    print(f"   • Date début : {args.start}")
    print(f"   • Date fin   : {args.end}")
    print(f"   • Sauvegarde : {'NON' if args.no_save else 'OUI'}")
    
    # Initialiser le générateur avec les dates
    generator = DataGenerator(start_date=args.start, end_date=args.end)
    
    # Générer les données (save=True sauf si --no-save)
    df = generator.generate(save=(not args.no_save))
    
    print(f"\n✅ Données générées avec succès !")
    print(f"   • Lignes créées : {len(df):,}")
    print(f"   • Mémoire utilisée : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # === ÉTAPE 3 : IMPORT DANS LA BASE DE DONNÉES (OPTIONNEL) ===
    if args.import_db:
        print("\n" + "="*70)
        print("🗄️ IMPORT DANS LA BASE DE DONNÉES")
        print("="*70)
        
        # Importer la fonction d'initialisation
        from src.database import init_database
        
        # Initialiser la BD (drop_existing=True supprime l'ancienne si --drop-db)
        if args.drop_db:
            print("\n⚠️ ATTENTION : Suppression de la base de données existante...")
            confirmation = input("   Confirmer ? (oui/non) : ")
            if confirmation.lower() != 'oui':
                print("   ❌ Import annulé.")
                return df
        
        db = init_database(drop_existing=args.drop_db)
        
        if db:
            # Vérifier que le fichier CSV existe bien
            if RAW_DATA_FILE.exists():
                print(f"\n📁 Import du fichier : {RAW_DATA_FILE}")
                
                # Importer le CSV dans la BD
                # Cette fonction lit le CSV ligne par ligne et insère dans SQLite
                count = import_csv_to_db(RAW_DATA_FILE, db)
                
                if count > 0:
                    print(f"   ✅ {count:,} enregistrements importés")
                    
                    # === AFFICHER LES STATISTIQUES DE LA BD ===
                    print("\n📊 Statistiques de la base de données :")
                    stats = db.get_statistics()
                    
                    # Formater l'affichage des statistiques
                    for key, value in stats.items():
                        # Formater les nombres avec séparateurs de milliers
                        if isinstance(value, (int, float)):
                            if isinstance(value, float):
                                print(f"   • {key}: {value:.2f}")
                            else:
                                print(f"   • {key}: {value:,}")
                        else:
                            print(f"   • {key}: {value}")
                else:
                    print(f"   ⚠️ Aucun enregistrement importé (fichier vide ?)")
                
                # Fermer la connexion à la BD proprement
                db.close()
                print("\n✅ Connexion à la base de données fermée")
            else:
                print(f"\n❌ ERREUR : Fichier non trouvé : {RAW_DATA_FILE}")
                print("   💡 Relancez le script sans --no-save pour générer le CSV d'abord")
        else:
            print("\n❌ ERREUR : Impossible d'initialiser la base de données")
    
    # === ÉTAPE 4 : AFFICHER LE RÉSUMÉ FINAL ===
    print("\n" + "="*70)
    print("✅ SCRIPT TERMINÉ AVEC SUCCÈS")
    print("="*70)
    
    # Résumé des fichiers générés
    print(f"\n📁 FICHIERS GÉNÉRÉS :")
    if not args.no_save:
        print(f"   • CSV brut : {RAW_DATA_FILE}")
        print(f"     Taille : {RAW_DATA_FILE.stat().st_size / 1024**2:.2f} MB")
    else:
        print(f"   • Aucun fichier (mode --no-save)")
    
    if args.import_db:
        db_file = project_root / "data" / "power_outages.db"
        if db_file.exists():
            print(f"   • Base de données : {db_file}")
            print(f"     Taille : {db_file.stat().st_size / 1024**2:.2f} MB")
    
    # Statistiques du DataFrame
    print(f"\n📊 STATISTIQUES DES DONNÉES :")
    print(f"   • Nombre total d'enregistrements : {len(df):,}")
    print(f"   • Nombre de quartiers            : {df['quartier'].nunique()}")
    print(f"   • Période couverte               : {df['date_heure'].min()} → {df['date_heure'].max()}")
    print(f"   • Taux global de coupures        : {df['coupure'].mean()*100:.2f}%")
    
    # Répartition par quartier
    print(f"\n🏘️ RÉPARTITION PAR QUARTIER :")
    quartier_stats = df.groupby('quartier')['coupure'].agg(['count', 'mean'])
    quartier_stats.columns = ['Nb observations', 'Taux coupures']
    quartier_stats['Taux coupures'] = quartier_stats['Taux coupures'] * 100
    
    for quartier, row in quartier_stats.iterrows():
        print(f"   • {quartier:20s} : {row['Nb observations']:6,} obs, {row['Taux coupures']:5.2f}% coupures")
    
    # Avertissement si pas de sauvegarde
    if args.no_save:
        print(f"\n⚠️ ATTENTION : Données non sauvegardées (--no-save)")
        print(f"   Pour sauvegarder, relancez sans l'option --no-save")
    
    return df


# ============================================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================================

if __name__ == "__main__":
    """
    Point d'entrée quand on exécute : python scripts/1_generate_data.py
    
    Ce script est le PREMIER à exécuter dans le pipeline du projet.
    Sans données, les autres scripts (preprocess, train, evaluate) ne peuvent pas fonctionner.
    
    ORDRE D'EXÉCUTION DU PROJET :
    1. 🔵 python scripts/1_generate_data.py       ← VOUS ÊTES ICI
    2. 🟢 python scripts/2_train_models.py
    3. 🟡 python scripts/3_evaluate_models.py
    4. 🟠 python scripts/4_deploy_api.py (ou app.py)
    
    STRUCTURE DES DONNÉES GÉNÉRÉES :
    Le CSV généré contient ces colonnes :
    
    ┌─────────────┬──────────────┬─────────────────────────────────────┐
    │   Colonne   │     Type     │           Description               │
    ├─────────────┼──────────────┼─────────────────────────────────────┤
    │ date_heure  │ datetime     │ Timestamp (horaire)                 │
    │ quartier    │ str          │ Nom du quartier (6 quartiers)       │
    │ temperature │ float        │ Température en °C (25-40°C)         │
    │ humidite    │ float        │ Humidité en % (30-95%)              │
    │ vitesse_vent│ float        │ Vitesse du vent en km/h (0-50)      │
    │ pluie       │ int          │ Pluie ? (0=non, 1=oui)              │
    │ jour_semaine│ int          │ Jour de la semaine (0=lun, 6=dim)   │
    │ heure       │ int          │ Heure de la journée (0-23)          │
    │ mois        │ int          │ Mois (1-12)                         │
    │ coupure     │ int          │ Coupure ? (0=non, 1=oui) ← CIBLE   │
    └─────────────┴──────────────┴─────────────────────────────────────┘
    
    PATTERNS SIMULÉS DANS LES DONNÉES :
    1. Saisonnalité :
       - Plus de coupures en saison chaude (avril-juin)
       - Moins de coupures en saison fraîche (décembre-février)
    
    2. Heures de pointe :
       - Pics de coupures : 13h-15h et 20h-22h
       - Creux : 3h-5h (nuit)
    
    3. Différences entre quartiers :
       - Quartiers populaires (Guédiawaye, Pikine) : Plus de coupures
       - Quartiers résidentiels (Plateau, Almadies) : Moins de coupures
    
    4. Influence météo :
       - Chaleur extrême → Plus de coupures (climatisation)
       - Pluie → Plus de coupures (court-circuits)
       - Vent fort → Plus de coupures (lignes endommagées)
    
    TAILLE ATTENDUE DU DATASET :
    - 1 an de données horaires = 365 jours × 24h = 8,760 heures
    - 6 quartiers
    - Total : 8,760 × 6 = 52,560 lignes
    - Taille fichier : ~3-5 MB (CSV)
    
    RÉSOLUTION DES PROBLÈMES COURANTS :
    
    ❌ Problème : "ModuleNotFoundError: No module named 'src'"
    ✅ Solution : Vérifiez que vous êtes dans le dossier racine du projet
    
    ❌ Problème : "FileNotFoundError: data/raw/"
    ✅ Solution : Les dossiers sont créés automatiquement, mais vérifiez
                  que vous avez les droits d'écriture
    
    ❌ Problème : "PermissionError" lors de --drop-db
    ✅ Solution : Fermez tout programme qui utilise la BD (DB Browser, etc.)
    
    ❌ Problème : Les données semblent irréalistes
    ✅ Solution : C'est normal, ce sont des données synthétiques !
                  Ajustez les paramètres dans src/data_generator.py
    """
    main()


# ============================================================================
# NOTES PÉDAGOGIQUES POUR DATA SCIENTIST JUNIOR
# ============================================================================

"""
📚 CONCEPTS CLÉS À RETENIR :

1. POURQUOI GÉNÉRER DES DONNÉES SYNTHÉTIQUES ?
   --------------------------------------------
   En projet réel, vous auriez accès aux vraies données de SENELEC.
   Ici, on simule car :
   - Pas d'accès aux données réelles (confidentielles)
   - Permet de tester le pipeline complet
   - Contrôle total sur les patterns (pour tester les modèles)
   - Reproductibilité (mêmes données à chaque génération)

2. STRUCTURE D'UN BON SCRIPT DE GÉNÉRATION
   ----------------------------------------
   ✅ Arguments en ligne de commande (flexibilité)
   ✅ Validation des paramètres (dates, chemins)
   ✅ Messages informatifs (progression, statistiques)
   ✅ Gestion d'erreurs (try/except)
   ✅ Documentation complète (ce que vous lisez !)

3. BONNES PRATIQUES - GESTION DES CHEMINS
   ---------------------------------------
   Au lieu de :
     ❌ sys.path.append("../")  # Fragile !
   
   On utilise :
     ✅ Path(__file__).parent.parent  # Robuste !
   
   Pourquoi ? Cela fonctionne peu importe d'où vous lancez le script.

4. ARGUMENTS DE LIGNE DE COMMANDE (argparse)
   ------------------------------------------
   argparse est LA bibliothèque standard pour parser les arguments.
   
   Types d'arguments :
   - Positionnels : python script.py valeur
   - Optionnels : python script.py --option valeur
   - Flags (booléens) : python script.py --flag
   
   Dans notre script :
   - --start, --end : Optionnels avec valeur (dates)
   - --no-save, --import-db : Flags (juste présence/absence)

5. SÉPARATION DES RESPONSABILITÉS
   --------------------------------
   Ce script est un "orchestrateur" :
   - Il gère les arguments (interface utilisateur)
   - Il appelle DataGenerator (logique métier)
   - Il appelle DatabaseManager (persistance)
   
   Principe : "Une fonction = une responsabilité"
   
   ❌ MAUVAIS : Tout dans main() (15000 lignes)
   ✅ BON : main() orchestre, modules font le travail

6. GESTION DE LA BASE DE DONNÉES
   ------------------------------
   Option --import-db permet de stocker les données dans SQLite.
   
   Avantages SQLite :
   - Fichier unique (.db)
   - Pas de serveur à lancer
   - SQL standard (apprentissage)
   - Intégration facile avec Pandas
   
   Quand utiliser --drop-db ?
   - Changement de structure des données
   - Corruption de la BD
   - Régénération complète
   ⚠️ Attention : Supprime TOUT !

7. STATISTIQUES ET VALIDATION
   ---------------------------
   Toujours afficher des stats après génération :
   - Nombre de lignes (vérifier qu'on a tout)
   - Taux de coupures (cohérent avec l'attendu ?)
   - Répartition par quartier (équilibrée ?)
   - Période couverte (dates correctes ?)
   
   Si quelque chose semble bizarre, INVESTIGUER !

8. WORKFLOW TYPIQUE D'UTILISATION
   -------------------------------
   Première fois (setup complet) :
   1. python scripts/1_generate_data.py --import-db
   
   Régénération (changement de paramètres) :
   2. python scripts/1_generate_data.py --import-db --drop-db
   
   Test rapide (sans sauvegarder) :
   3. python scripts/1_generate_data.py --no-save
   
   Période personnalisée :
   4. python scripts/1_generate_data.py --start 2024-01-01 --end 2024-06-30

9. DÉBOGAGE COURANT
   -----------------
   Si le script plante :
   1. Vérifiez les messages d'erreur (lisez-les vraiment !)
   2. Vérifiez que vous êtes dans le bon dossier (racine du projet)
   3. Vérifiez que les dossiers data/ existent
   4. Essayez avec --no-save d'abord (test sans sauvegarde)
   5. Vérifiez les imports (pip install -r requirements.txt)

10. COMMANDES UTILES
    -----------------
    # Génération standard
    python scripts/1_generate_data.py
    
    # Voir les données générées
    head data/raw/power_outages.csv
    
    # Compter les lignes
    wc -l data/raw/power_outages.csv
    
    # Vérifier la taille
    ls -lh data/raw/power_outages.csv
    
    # Ouvrir avec pandas (Python)
    import pandas as pd
    df = pd.read_csv('data/raw/power_outages.csv')
    df.info()
"""