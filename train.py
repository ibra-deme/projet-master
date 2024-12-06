import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r',encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
# import numpy as np
# import random
# import json
# import os

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertModel

# from model import NeuralNet

# # Charger les intents
# try:
#     with open('intents.json', 'r', encoding='utf-8') as f:
#         intents = json.load(f)
# except (FileNotFoundError, json.JSONDecodeError) as e:
#     print("Erreur lors du chargement de 'intents.json':", e)
#     exit()

# tags = []
# xy = []
# all_words = []  # Liste des mots uniques

# # Boucle à travers les intents
# for intent in intents.get('intents', []):
#     tag = intent.get('tag')
#     if not tag:
#         continue
#     tags.append(tag)
#     for pattern in intent.get('patterns', []):
#         xy.append((pattern, tag))
#         # Tokenisation et ajout des mots uniques
#         words = pattern.split()  # Tokenisation simple (ou utilisez NLTK pour une meilleure tokenisation)
#         all_words.extend(words)

# if not xy:
#     print("Aucun pattern trouvé dans 'intents.json'.")
#     exit()

# # Garder uniquement les mots uniques
# all_words = sorted(list(set(all_words)))

# # Charger BERT
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')

# def bert_embedding(sentence):
#     """
#     Génère les embeddings BERT pour une phrase.
#     """
#     inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
#     outputs = bert_model(**inputs)
#     cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [1, hidden_size]
#     return cls_embedding.detach().numpy().squeeze()

# # Création des données d'entraînement
# X = []
# y = []
# for (pattern_sentence, tag) in xy:
#     embedding = bert_embedding(pattern_sentence)
#     X.append(embedding)
#     label = tags.index(tag)
#     y.append(label)

# X = np.array(X)
# y = np.array(y)

# # Séparation entraînement/validation
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Paramètres
# num_epochs = 500
# batch_size = 16
# learning_rate = 0.001
# input_size = 768  # Taille des embeddings BERT
# hidden_size = 16
# output_size = len(tags)

# print(f"Entraînement: {len(X_train)} exemples, Validation: {len(X_val)} exemples")
# print(input_size, output_size)

# class ChatDataset(Dataset):
#     def __init__(self, X, y):
#         self.n_samples = len(X)
#         self.x_data = torch.tensor(X, dtype=torch.float32)
#         self.y_data = torch.tensor(y, dtype=torch.long)

#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     def __len__(self):
#         return self.n_samples

# train_dataset = ChatDataset(X_train, y_train)
# val_dataset = ChatDataset(X_val, y_val)

# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = NeuralNet(input_size, hidden_size, output_size).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Entraînement
# for epoch in range(num_epochs):
#     model.train()
#     for words, labels in train_loader:
#         words = words.to(device)
#         labels = labels.to(device)

#         outputs = model(words)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # Validation
#     if (epoch + 1) % 50 == 0:
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for words, labels in val_loader:
#                 words = words.to(device)
#                 labels = labels.to(device)
#                 outputs = model(words)
#                 val_loss += criterion(outputs, labels).item()

#         print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

# # Sauvegarde
# data = {
#     "model_state": model.state_dict(),
#     "input_size": input_size,
#     "hidden_size": hidden_size,
#     "output_size": output_size,
#     "tags": tags,
#     "all_words": all_words  # Sauvegarde de all_words
# }

# FILE = "data.pth"
# torch.save(data, FILE)
# print(f"Entraînement terminé. Modèle sauvegardé dans {FILE}.")
