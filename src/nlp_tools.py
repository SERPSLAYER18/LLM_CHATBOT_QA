import spacy
from nltk.corpus import stopwords
import json
import re
import string

nlp = spacy.load("ru_core_news_md")
stop_words = stopwords.words("russian") + ["—"]


def lemmatize_russian(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]


def remove_stopwords_and_punct(lemmas):
    return [
        lemma
        for lemma in lemmas
        if not (lemma in stop_words or lemma in string.punctuation)
    ]


def tokenize(text):
    lemmas = lemmatize_russian(text)
    tokens = remove_stopwords_and_punct(lemmas)
    return tokens
