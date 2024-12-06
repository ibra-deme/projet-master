# Importer toutes les fonctions du fichier
from nltk_utils import correct_spelling_spellchecker, tokenize, stem, bag_of_words

def test_correct_spelling():
    sentence = "je viens d'avoir mon bac"
    corrected = correct_spelling_spellchecker(sentence)
    print(f"Phrase originale : {sentence}")
    print(f"Phrase corrigée : {corrected}")

def test_tokenize():
    sentence = "je viens d'avoir mon bac!"
    tokens = tokenize(sentence)
    print(f"Tokens : {tokens}")

def test_stem():
    words = ["organize", "organizes", "organizing"]
    stemmed_words = [stem(word) for word in words]
    print(f"Mots originaux : {words}")
    print(f"Mots après stemming : {stemmed_words}")

def test_bag_of_words():
    sentence = "je viens d'avoir mon bac"
    words = ["ceci", "test", "bag", "word"]
    tokens = tokenize(sentence)
    bag = bag_of_words(tokens, words)
    print(f"Sac de mots : {bag}")

if __name__ == "__main__":
    test_correct_spelling()
    test_tokenize()
    test_stem()
    test_bag_of_words()
