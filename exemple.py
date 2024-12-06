# Entraînement du modèle
for epoch in range(10):  # boucle d'apprentissage sur 10 époques
    for data, targets in train_loader:
        optimizer.zero_grad()  # Réinitialiser les gradients
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()  # Rétropropagation
        optimizer.step()  # Mise à jour des poids
