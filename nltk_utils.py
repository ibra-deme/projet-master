# import numpy as np
# import nltk
# import symspellpy
# from symspellpy.symspellpy import SymSpell, Verbosity
# # nltk.download('punkt')
# from textblob import TextBlob
# from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()

# def correct_spelling(sentence):
#     """
#     Correct spelling in the input sentence.
#     """
#     corrected = TextBlob(sentence).correct()
#     return str(corrected)

# def tokenize(sentence):
#     """
#     split sentence into array of words/tokens
#     a token can be a word or punctuation character, or number
#     """
#     return nltk.word_tokenize(sentence)


# def stem(word):
#     """
#     stemming = find the root form of the word
#     examples:
#     words = ["organize", "organizes", "organizing"]
#     words = [stem(w) for w in words]
#     -> ["organ", "organ", "organ"]
#     """
#     return stemmer.stem(word.lower())


# def bag_of_words(tokenized_sentence, words):
#     """
#     Retourne un sac de mots avec correction orthographique et normalisation.
#     """
#     # Corriger l'orthographe avant de traiter
#     corrected_sentence = correct_spelling(" ".join(tokenized_sentence))
#     corrected_tokens = tokenize(corrected_sentence)

#     # Stemmatiser chaque mot
#     sentence_words = [stem(word) for word in corrected_tokens]
    
#     # Initialiser le sac de mots
#     bag = np.zeros(len(words), dtype=np.float32)
#     for idx, w in enumerate(words):
#         if w in sentence_words: 
#             bag[idx] = 1

#     return bag

#avant
import numpy as np
import nltk

from textblob import TextBlob
from nltk.stem.porter import PorterStemmer
from spellchecker import SpellChecker

stemmer = PorterStemmer()

# Charger le modèle français de SpaCy
# nlp = spacy.load('fr_core_news_sm')

# def correct_spelling_spacy(sentence):
#     """
#     Utilise SpaCy pour le traitement de la phrase et retourne la phrase corrigée.
#     Cette fonction améliore la structure de la phrase et peut être utilisée pour la correction syntaxique,
#     mais elle n'est pas un correcteur orthographique à proprement parler.
#     """
#     doc = nlp(sentence)
#     corrected_sentence = " ".join([token.text for token in doc])
#     return corrected_sentence


# def correct_spelling_textblob(sentence):
#     """
#     Correcte l'orthographe dans la phrase avec TextBlob.
#     Utilisé pour les corrections simples.
#     """
#     blob = TextBlob(sentence)
#     return str(blob.correct())
def correct_spelling_spellchecker(sentence):
    spell = SpellChecker(language='fr')  # Pour corriger en français
    words = sentence.split()  # Séparer la phrase en mots
    corrected_words = [spell.correction(word) for word in words]  # Correction des fautes
    corrected_sentence = " ".join(corrected_words)  # Rejoindre les mots corrigés
    return corrected_sentence

def tokenize(sentence):
    """
    Divise une phrase en une liste de tokens.
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    Effectue un stemming (réduction des mots à leur racine).
    """
    return stemmer.stem(word.lower())


# def bag_of_words(tokenized_sentence, words):
#     """
#     Crée un sac de mots (bag of words) après correction orthographique et stemming.
#     """
#     # Correction orthographique de la phrase avant traitement
#     corrected_sentence = correct_spelling_spacy(" ".join(tokenized_sentence))
#     corrected_tokens = tokenize(corrected_sentence)

#     # Stemmatisation de chaque mot
#     sentence_words = [stem(word) for word in corrected_tokens]

#     # Initialisation du sac de mots
#     bag = np.zeros(len(words), dtype=np.float32)
#     for idx, w in enumerate(words):
#         if w in sentence_words:
#             bag[idx] = 1

#     return bag
def bag_of_words(tokenized_sentence, words):
    """
    Crée un sac de mots (bag of words) après correction orthographique et stemming.
    """
    # Correction orthographique de la phrase avant traitement
    corrected_sentence = correct_spelling_spellchecker(" ".join(tokenized_sentence))
    print("Corrected Sentence:", corrected_sentence)  # Affichage de la phrase corrigée
    corrected_tokens = tokenize(corrected_sentence)
    print("Tokens après correction:", corrected_tokens)  # Affichage des tokens après correction

    # Stemmatisation de chaque mot
    sentence_words = [stem(word) for word in corrected_tokens]
    print("Tokens après stemming:", sentence_words)  # Affichage des tokens après stemming

    # Initialisation du sac de mots
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    print("Sac de mots:", bag)  # Affichage du sac de mots
    return bag

