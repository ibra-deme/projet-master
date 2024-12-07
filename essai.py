import sqlite3

def get_db_connection():
    try:
        connection = sqlite3.connect("chatbot_db.sqlite")
        connection.execute("PRAGMA foreign_keys = 1")  # Assure l'intégrité référentielle
        return connection
    except sqlite3.Error as e:
        print(f"Erreur lors de la connexion à la base de données : {e}")
        return None


