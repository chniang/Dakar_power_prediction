# Type de base de données
DATABASE_TYPE = 'mysql'  # ← Changer de 'sqlite' à 'mysql'

# MySQL (pour production)
MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'database': 'dakar_predictions',  # ← TON NOM DE BD
    'user': 'root',
    'password': '',  # ← Pas de mot de passe
    'charset': 'utf8mb4'
}