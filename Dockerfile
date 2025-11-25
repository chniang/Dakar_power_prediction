# Utiliser une image Python de base
        FROM python:3.10-slim

        # Définir le répertoire de travail
        WORKDIR /app

        # Installer les dépendances
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt

        # Copier le reste du code
        COPY . .

        # Définir la commande de démarrage (important : Streamlit doit tourner sur le port 7860)
        CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port", "7860", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]