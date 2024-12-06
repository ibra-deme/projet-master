# import random
# import json

# import torch

# from model import NeuralNet
# from nltk_utils import bag_of_words, tokenize

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

# bot_name = "El-tocarico"
# print("Let's chat! (type 'quit' to exit)")
# while True:
#     # sentence = "do you use credit cards?"
#     sentence = input("You: ")
#     if sentence == "quit":
#         break

#     sentence = tokenize(sentence)
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
#                 print(f"{bot_name}: {random.choice(intent['responses'])}")
#     else:
#         print(f"{bot_name}: I do not understand...")


import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Charger les intents
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Charger les données du modèle
FILE = "data.pth"
data = torch.load(FILE)

input_size = 768
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialiser le modèle
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "El-tocarico"
chat_history = []  # Historique des échanges pour conserver le contexte

print("Let's chat! (type 'quit' to exit)")

while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    # Tokenisation et conversion en bag-of-words
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Prédiction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Calcul de la probabilité
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        # Réponse pour un intent identifié avec une haute confiance
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                chat_history.append({"user": sentence, "bot": response})
                print(f"{bot_name}: {response}")
    else:
        # Réponse pour un intent avec une faible confiance
        suggestions = [intent["tag"] for intent in intents['intents'][:3]]  # Top 3 suggestions
        print(f"{bot_name}: I'm not sure. Did you mean:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")
        print(f"{bot_name}: Please clarify your question.")

# import random
# import json
# import torch
# from model import NeuralNet
# from nltk_utils import bert_similarity  # Utilisation de la fonction pour calculer la similarité avec BERT
# from transformers import BertTokenizer, BertModel

# # Charger le tokenizer et le modèle BERT pré-entraîné
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Charger les intents
# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# # Charger les données du modèle
# FILE = "data.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# # Initialiser le modèle
# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# bot_name = "El-tocarico"
# chat_history = []  # Historique des échanges pour conserver le contexte

# print("Let's chat! (type 'quit' to exit)")

# while True:
#     sentence = input("You: ")
#     if sentence.lower() == "quit":
#         break

#     # Calculer la similarité avec les modèles BERT
#     max_similarity = 0
#     best_intent = None

#     for intent in intents['intents']:
#         for pattern in intent['patterns']:
#             similarity = bert_similarity(sentence, pattern)  # Utilisation de la fonction de similarité de BERT
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 best_intent = intent

#     # Si la similarité est élevée, répondre avec l'intent correspondant
#     if max_similarity > 0.75:
#         response = random.choice(best_intent['responses'])
#         chat_history.append({"user": sentence, "bot": response})
#         print(f"{bot_name}: {response}")
#     else:
#         # Réponse pour une faible similarité, avec des suggestions
#         suggestions = [intent["tag"] for intent in intents['intents'][:3]]  # Top 3 suggestions
#         print(f"{bot_name}: I'm not sure. Did you mean:")
#         for suggestion in suggestions:
#             print(f"  - {suggestion}")
#         print(f"{bot_name}: Please clarify your question.")
