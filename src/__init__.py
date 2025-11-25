# Type de base de données
DATABASE_TYPE = 'mysql'  # ← Changer de 'sqlite' à 'mysql'

# MySQL (pour production)
MYSQL_CONFIG = {
    'host': 'localhost',           # Serveur local (ou IP distante)
    'port': 3306,                  # Port par défaut MySQL
    'database': 'dakar_predictions',
    'user': 'root',
    'password': '',                # ← Vide en dev (à sécuriser en prod !)
    'charset': 'utf8mb4'           # Encodage UTF-8 complet
}
