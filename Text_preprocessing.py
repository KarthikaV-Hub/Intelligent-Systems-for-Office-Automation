import os, re, io, random
import numpy as np
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

sample_text = """Thiss   is an exampel  text corpuz!!!
It contains many  ERRORS,    extra spaces, strange punctuation... and we need to clean it.
Also it contains abbreviations e.g. and acronyms (NLP).Let's see sentence boundaries... maybe not perfect?"""

with open("file.txt","w", encoding="utf8") as f:
    f.write(sample_text)

with open("file.txt","r", encoding="utf8") as f:
    corpus = f.read()

print("loaded corpus:\n", corpus, "\n")

tokens = word_tokenize(corpus)
print("First 30 tokens:", tokens[:30])

corrected_tokens = []
for t in tokens:
    if re.fullmatch(r'\W+', t) or re.fullmatch(r'\d+', t):
        corrected_tokens.append(t)
    else:
        try:
            corrected_tokens.append(str(TextBlob(t).correct()))
        except:
            corrected_tokens.append(t)

print("\nFirst 10 corrected tokens:", corrected_tokens[:10])
corrected_text = " ".join(corrected_tokens)
print("\nCorrected text corpus:\n", corrected_text)

pos_tags = nltk.pos_tag(corrected_tokens)
print("\nPOS tags (first 40):\n", pos_tags[:40])

stopset = set(stopwords.words('english'))
filtered_tokens = [w for w in corrected_tokens if w.lower() not in stopset and re.search(r'\w', w)]
print("\nFirst 20 tokens after stopword removal:\n", filtered_tokens[:20])

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stems = [stemmer.stem(w) for w in filtered_tokens]
lemmas = [lemmatizer.lemmatize(w) for w in filtered_tokens]

print("\nFirst 20 stems:\n", stems[:20])
print("\nFirst 20 lemmas:\n", lemmas[:20])

sentences = sent_tokenize(corpus)
print("\nDetected sentences (count):", len(sentences))
print("Sentences:\n", sentences)
