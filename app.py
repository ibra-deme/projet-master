#Premiere partie sans bd
# from flask import Flask, request, jsonify
# import torch
# import json
# import random
# from model import NeuralNet
# from flask_cors import CORS
# from nltk_utils import bag_of_words, tokenize

# app = Flask(__name__)
# CORS(app)  # Activer CORS

# # Charger le modèle
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# FILE = "data.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# def generate_response(user_message):
#     # Tokeniser le message de l'utilisateur
#     sentence = tokenize(user_message)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     # Obtenir la réponse du modèle
#     output = model(X)
#     _, predicted = torch.max(output, dim=1)
#     tag = tags[predicted.item()]

#     # Vérifier la probabilité
#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 return random.choice(intent['responses'])
#     else:
#         return "Je ne comprends pas..."

# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json()
#     user_message = data.get("message")
#     response = generate_response(user_message)
#     return jsonify(response=response)  # Correction ici

# if __name__ == "__main__":
#     app.run(debug=True)

#Deuxieme partie avec mysql
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import mysql.connector
# import torch
# import json
# import random
# from model import NeuralNet
# from nltk_utils import bag_of_words, tokenize
# from werkzeug.security import generate_password_hash, check_password_hash

# app = Flask(__name__)
# CORS(app)

# # Configuration de MySQL
# def get_db_connection():
#     try:
#         connection = mysql.connector.connect(
#             host="localhost",
#             user="root",
#             password="",
#             database="chatbot_db"
#         )
#         if connection.is_connected():
#             print("La connexion à la base de données a été établie avec succès.")
#         return connection
#     except mysql.connector.Error as e:
#         print(f"Erreur lors de la connexion à la base de données : {e}")
#         return None
# get_db_connection()
# # Charger le modèle
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# FILE = "data.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# def generate_response(user_message):
#     sentence = tokenize(user_message)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)
#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 return random.choice(intent['responses'])
#     else:
#         return "Je ne comprends pas..."
# @app.route("/", methods=["GET"])
# def home():
#     connection = get_db_connection()
#     if connection:
#         return jsonify(message="Le serveur fonctionne correctement et la connexion à la base de données a été établie avec succès."), 200
#     else:
#         return jsonify(message="Échec de la connexion à la base de données."), 500

# # Enregistrement utilisateur
# @app.route("/register", methods=["POST"])
# def register():
#     data = request.get_json()
#     prenom = data.get("prenom")
#     nom = data.get("nom")
#     email = data.get("email")
#     password = data.get("password")
    
#     if not email or '@' not in email:
#         return jsonify(message="Invalid email address"), 400
    
#     password_hash = generate_password_hash(password)
    
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     try:
#         cursor.execute(
#             "INSERT INTO users (prenom, nom, email, password) VALUES (%s, %s, %s, %s)", 
#             (prenom, nom, email, password_hash)
#         )
#         conn.commit()
#         return jsonify(message="User registered successfully")
#     except Exception as e:
#         conn.rollback()
#         return jsonify(message="An error occurred", error=str(e)), 400
#     finally:
#         cursor.close()
#         conn.close()

# # Connexion utilisateur
# @app.route("/login", methods=["POST"])
# def login():
#     data = request.get_json()
#     email = data.get("email")
#     password = data.get("password")
    
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
#     user = cursor.fetchone()
#     cursor.close()
#     conn.close()
    
#     if user and check_password_hash(user[3], password):
#         return jsonify(message="Login successful", user_id=user[0])
#     else:
#         return jsonify(message="Invalid email or password"), 401

# # Endpoint de chat avec enregistrement de la conversation
# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json()
#     user_message = data.get("message")
#     user_id = data.get("user_id")

#     if not user_message:
#         return jsonify(message="Empty message"), 400

#     response = generate_response(user_message)
    
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute(
#         "INSERT INTO chats (user_id, message, response) VALUES (%s, %s, %s)",
#         (user_id, user_message, response)
#     )
#     conn.commit()
#     cursor.close()
#     conn.close()

#     return jsonify(response=response)

# if __name__ == "__main__":
#     app.run(debug=True)
#Avant utilisation de BERT

from fuzzywuzzy import process  
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import torch
import json
import random
from model import NeuralNet
from nltk_utils import correct_spelling_spellchecker, tokenize, bag_of_words

from nltk_utils import bag_of_words, tokenize
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

# Configuration de SQLite
def get_db_connection():
    try:
        connection = sqlite3.connect("my_chatbot.sqlite")
        connection.execute("PRAGMA journal_mode = WAL;")  # Active le mode WAL
        connection.execute("PRAGMA foreign_keys = 1;") 
       
        return connection
        # conn = sqlite3.connect("chatbot_db.sqlite")
        # conn.execute("PRAGMA journal_mode = WAL;")  # Active le mode WAL
        # conn.execute("PRAGMA foreign_keys = 1")
    except sqlite3.Error as e:
        print(f"Erreur lors de la connexion à la base de données : {e}")
        return None

# Charger le modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r',encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# def generate_response(user_message):
#     sentence = tokenize(user_message)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)
#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 return random.choice(intent['responses'])
#     else:
#         return "Je ne comprends pas..."
def generate_response(user_message):
    # Correction orthographique du message de l'utilisateur
    corrected_message = correct_spelling_spellchecker(user_message)
    print("Message corrigé:", corrected_message)
    
    # Tokenisation du message corrigé
    sentence = tokenize(corrected_message)
    print("Tokens:", sentence)

    # Création du sac de mots
    X = bag_of_words(sentence, all_words)
    print("Sac de mots:", X)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Passer le message corrigé dans le modèle
    output = model(X)
    print("Sortie du modèle:", output)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    print("Tag prédit:", tag)

    # Calcul de la probabilité
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()] if probs.size(1) > 0 else 0.0
    print(f"Probabilité: {prob.item()}")

    # Vérification de la probabilité de la réponse
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print("Réponse choisie:", response)
                return response
    else:
        print("Probabilité insuffisante pour une réponse valide.")
        return "Je ne comprends pas..."





@app.route("/", methods=["GET"])
def home():
    connection = get_db_connection()
    if connection:
        return jsonify(message="Le serveur fonctionne correctement et la connexion à la base de données SQLite a été établie avec succès."), 200
    else:
        return jsonify(message="Échec de la connexion à la base de données."), 500

# Enregistrement utilisateur
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    prenom = data.get("prenom")
    nom = data.get("nom")
    email = data.get("email")
    password = data.get("password")
    
    if not email or '@' not in email:
        return jsonify(message="Invalid email address"), 400
    
    password_hash = generate_password_hash(password)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (prenom, nom, email, password) VALUES (?, ?, ?, ?)", 
            (prenom, nom, email, password_hash)
        )
        conn.commit()
        return jsonify(message="User registered successfully")
    except Exception as e:
        conn.rollback()
        return jsonify(message="An error occurred", error=str(e)), 400
    finally:
        cursor.close()
        conn.close()

# Connexion utilisateur
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if user:
        # Affichez le hash pour vérifier ce qui est stocké
        print("Mot de passe hashé stocké:", user[3])
        print("Mot de passe reçu:", password)
        
        if check_password_hash(user[4], password):
            return jsonify(message="Login successful", user_id=user[0])
        else:
            return jsonify(message="Invalid password"), 401
    else:
        return jsonify(message="User not found"), 404


# Endpoint de chat avec enregistrement de la conversation
@app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json()
#     user_message = data.get("message", "")
#     user_id = data.get("user_id", 0)
    
#     try:
#         # Connexion à la base de données avec un timeout
#         conn = sqlite3.connect("my_chatbot.sqlite", timeout=10)
#         cursor = conn.cursor()
        
#         # Effectuer une requête
#         cursor.execute("INSERT INTO chats (user_id, message) VALUES (?, ?)", (user_id, user_message))
#         conn.commit()
        
#         # Exécuter une autre requête ou retourner une réponse
#         response = {"response": "Message reçu !"}
#         return jsonify(response)
#     except sqlite3.OperationalError as e:
#         # Gestion des erreurs
#         return jsonify({"error": "Database is locked", "details": str(e)}), 500
#     finally:
#         cursor.close()
#         conn.close()
def chat():
    data = request.get_json()
    user_message = data.get("message")
    user_id = data.get("user_id")

    if not user_message:
        return jsonify(message="Empty message"), 400

    response = generate_response(user_message)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chats (user_id, message, response) VALUES (?, ?, ?)",
        (user_id, user_message, response)
    )
    conn.commit()
    cursor.close()
    conn.close()

    # Ajoutez une réponse de confirmation ici
    return jsonify(message="Message processed successfully", response=response), 200

@app.route("/messages", methods=["GET"])
def get_messages():
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify(message="User ID is required"), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT message, response, timestamp FROM chats WHERE user_id = ? ORDER BY timestamp ASC", 
        (user_id,)
    )
    messages = cursor.fetchall()
    cursor.close()
    conn.close()

    # Formater les messages dans un format plus pratique pour le frontend
    messages_list = [{'message': msg[0], 'response': msg[1], 'timestamp': msg[2]} for msg in messages]
    
    return jsonify(messages=messages_list)


if __name__ == "__main__":
    app.run(debug=True)
 
# from fuzzywuzzy import process  
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import sqlite3
# import torch
# import json
# import random
# from nltk_utils import correct_spelling, tokenize, bag_of_words, bert_similarity

# from model import NeuralNet
# from nltk_utils import correct_spelling, tokenize, bag_of_words
# from werkzeug.security import generate_password_hash, check_password_hash

# app = Flask(__name__)
# CORS(app)

# # Configuration de SQLite
# def get_db_connection():
#     try:
#         connection = sqlite3.connect("chatbot_db.sqlite")
#         connection.execute("PRAGMA foreign_keys = 1")  # Assure l'intégrité référentielle
#         return connection
#     except sqlite3.Error as e:
#         print(f"Erreur lors de la connexion à la base de données : {e}")
#         return None

# # Charger le modèle
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# with open('intents.json', 'r', encoding='utf-8') as json_data:
#     intents = json.load(json_data)

# FILE = "data.pth"
# data = torch.load(FILE)

# # Extraire les hyperparamètres et autres données
# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# model_state = data["model_state"]
# all_words = data["all_words"]
# tags = data["tags"]

# # Recréer le modèle avec les bonnes dimensions
# model = NeuralNet(input_size, hidden_size, output_size)
# model.load_state_dict(model_state)
# model.eval()

# print("Modèle chargé avec succès.")

# all_words = sorted(list(set(all_words)))  # S'assurer que tous les mots sont uniques

# # Met à jour l'input_size pour correspondre à la taille de 'all_words'
# # input_size = len(all_words)

# # # Charger le modèle avec la bonne taille d'entrée
# # model = NeuralNet(input_size, hidden_size, output_size).to(device)
# # model.load_state_dict(model_state)
# # model.eval()
# print(f"Input size in model: {input_size}")
# print(f"Actual input size from all_words: {len(all_words)}")

# def generate_response(user_message):
#     # Tokeniser le message utilisateur et corriger les fautes d'orthographe si nécessaire
#     sentence = tokenize(user_message)
#     sentence = [correct_spelling(word) for word in sentence]  # Optionnel, si vous voulez corriger l'orthographe

#     # Convertir la phrase en bag of words
#     X = bag_of_words(sentence, all_words)
#     if X is not None:
#        X = X.reshape(1, -1)
#     else:
#       print("Error: X is None")
#       return "An error occurred during processing."
#     X = torch.from_numpy(X).to(device)

#     # Prédire la catégorie de la phrase
#     output = model(X)
#     _, predicted = torch.max(output, dim=1)
#     tag = tags[predicted.item()]

#     # Calculer la probabilité de la prédiction
#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
    
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 # Ajouter une logique de similarité avec BERT
#                 best_response = None
#                 best_similarity = -1  # Pour stocker la similarité maximale

#                 for response in intent['responses']:
#                     similarity = bert_similarity(user_message, response)
#                     if similarity > best_similarity:
#                         best_similarity = similarity
#                         best_response = response

#                 return best_response if best_response else random.choice(intent['responses'])
#     else:
#         return "Je ne comprends pas..."


# @app.route("/", methods=["GET"])
# def home():
#     connection = get_db_connection()
#     if connection:
#         return jsonify(message="Le serveur fonctionne correctement et la connexion à la base de données SQLite a été établie avec succès."), 200
#     else:
#         return jsonify(message="Échec de la connexion à la base de données."), 500

# # Enregistrement utilisateur
# @app.route("/register", methods=["POST"])
# def register():
#     data = request.get_json()
#     prenom = data.get("prenom")
#     nom = data.get("nom")
#     email = data.get("email")
#     password = data.get("password")
    
#     if not email or '@' not in email:
#         return jsonify(message="Invalid email address"), 400
    
#     password_hash = generate_password_hash(password)
    
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     try:
#         cursor.execute(
#             "INSERT INTO users (prenom, nom, email, password) VALUES (?, ?, ?, ?)", 
#             (prenom, nom, email, password_hash)
#         )
#         conn.commit()
#         return jsonify(message="User registered successfully")
#     except Exception as e:
#         conn.rollback()
#         return jsonify(message="An error occurred", error=str(e)), 400
#     finally:
#         cursor.close()
#         conn.close()

# # Connexion utilisateur
# @app.route("/login", methods=["POST"])
# def login():
#     data = request.get_json()
#     email = data.get("email")
#     password = data.get("password")
    
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
#     user = cursor.fetchone()
#     cursor.close()
#     conn.close()
    
#     if user:
#         if check_password_hash(user[4], password):
#             return jsonify(message="Login successful", user_id=user[0])
#         else:
#             return jsonify(message="Invalid password"), 401
#     else:
#         return jsonify(message="User not found"), 404


# # Endpoint de chat avec enregistrement de la conversation
# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json()
#     user_message = data.get("message")
#     user_id = data.get("user_id")

#     if not user_message:
#         return jsonify(message="Empty message"), 400

#     response = generate_response(user_message)

#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute(
#         "INSERT INTO chats (user_id, message, response) VALUES (?, ?, ?)",
#         (user_id, user_message, response)
#     )
#     conn.commit()
#     cursor.close()
#     conn.close()

#     return jsonify(response=response)

# @app.route("/messages", methods=["GET"])
# def get_messages():
#     user_id = request.args.get('user_id')
    
#     if not user_id:
#         return jsonify(message="User ID is required"), 400

#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute(
#         "SELECT message, response, timestamp FROM chats WHERE user_id = ? ORDER BY timestamp ASC", 
#         (user_id,)
#     )
#     messages = cursor.fetchall()
#     cursor.close()
#     conn.close()

#     # Formater les messages dans un format plus pratique pour le frontend
#     messages_list = [{'message': msg[0], 'response': msg[1], 'timestamp': msg[2]} for msg in messages]
    
#     return jsonify(messages=messages_list)


# if __name__ == "__main__":
#     app.run(debug=True)
