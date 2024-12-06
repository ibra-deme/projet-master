import sqlite3

def get_db_connection():
    try:
        connection = sqlite3.connect("chatbot_db.sqlite")
        connection.execute("PRAGMA foreign_keys = 1")  # Assure l'intégrité référentielle
        return connection
    except sqlite3.Error as e:
        print(f"Erreur lors de la connexion à la base de données : {e}")
        return None

def modify_chats_table():
    # Obtenir la connexion
    conn = get_db_connection()
    if conn is None:
        print("Connexion à la base de données échouée.")
        return
    
    try:
        # Créer un curseur pour exécuter les commandes SQL
        cursor = conn.cursor()
        
        # Étape 1 : Ajouter la colonne 'timestamp' sans valeur par défaut
        cursor.execute("ALTER TABLE chats ADD COLUMN timestamp DATETIME;")
        
        # Étape 2 : Mettre à jour les lignes existantes avec CURRENT_TIMESTAMP
        cursor.execute("UPDATE chats SET timestamp = CURRENT_TIMESTAMP WHERE timestamp IS NULL;")
        
        # Confirmer les changements
        conn.commit()
        print("Table 'chats' modifiée avec succès.")
    except sqlite3.Error as e:
        print(f"Erreur lors de la modification de la table 'chats' : {e}")
    finally:
        # Fermer la connexion
        conn.close()

# Appel de la fonction pour modifier la table
modify_chats_table()
