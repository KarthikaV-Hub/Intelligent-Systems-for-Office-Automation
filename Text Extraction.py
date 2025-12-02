#Extraction of text features using
#BoW, N Gram, TF IDF, NER, Word2VEC, Glove and Fast Text

text = ["Karthi don't give up"]
print(text)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X_bow = cv.fit_transform(text)
print(cv.get_feature_names_out())
print(X_bow.toarray())

cv_ngram = CountVectorizer(ngram_range=(2, 2))
X_ngram = cv_ngram.fit_transform(text)
print(cv_ngram.get_feature_names_out())
print(X_ngram.toarray())

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(text)
print(tfidf.get_feature_names_out())
print(X_tfidf.toarray())

import spacy

nlp = spacy.load("en_core_web_sm")
sentence = "Karthi don't give up"
doc = nlp(sentence)

for ent in doc.ents:
    print(ent.text, ent.label_)

from gensim.models import Word2Vec

sentences = [["karthi", "dont", "give", "up"]]

w2v = Word2Vec(sentences, vector_size=5, window=2, min_count=1)
print(w2v.wv["give"])
print(len(w2v.wv["give"]))

import os
import urllib.request
import zipfile
import numpy as np

glove_filename = "glove.6B.50d.txt"
glove_zip = "glove.6B.zip"
glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"

def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
    print(f"\rDownloading GloVe: {percent:.1f}%", end="")

if not os.path.exists(glove_filename):
    try:
        if not os.path.exists(glove_zip):
            print("Downloading GloVe...")
            urllib.request.urlretrieve(glove_url, glove_zip, reporthook=_progress)
            print("\nDownload complete.")
        with zipfile.ZipFile(glove_zip, "r") as z:
            z.extract(glove_filename)
            print("GloVe extracted.")
    except Exception as e:
        print("GloVe download failed:", e)

glove = {}
if os.path.exists(glove_filename):
    with open(glove_filename, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            glove[word] = vector

    if "give" in glove:
        print(glove["give"])
        print(len(glove["give"]))
    else:
        print("Word not in GloVe vocab")
else:
    print("GloVe file still missing")

from gensim.models import FastText

ft = FastText(sentences, vector_size=5, window=2, min_count=1)
print(ft.wv["give"])
print(len(ft.wv["give"]))
