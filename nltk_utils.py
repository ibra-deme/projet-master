
#avant
import numpy as np
import nltk
import spellchecker
from spellchecker import SpellChecker
from textblob import TextBlob
from nltk.stem.porter import PorterStemmer
from spellchecker import SpellChecker

stemmer = PorterStemmer()


def correct_spelling_spellchecker(sentence):
    # Crée une instance de SpellChecker pour la langue française
    spell = SpellChecker(language='fr')
    corrected_words = []
    for word in sentence.split():
        corrected_word = spell.correction(word)
        corrected_words.append(corrected_word if corrected_word else word)
    
    corrected_sentence = " ".join(corrected_words)
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

