import pandas as pd
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

d = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
df = pd.DataFrame({'txt': d.data, 'cat': d.target})
df['txt'] = df['txt'].apply(lambda x: re.sub(r'[^a-zA-Z ]', ' ', x.lower()))

cv = CountVectorizer(stop_words='english', max_features=2000)
bow = cv.fit_transform(df['txt'])
b_sum = bow.toarray().sum(axis=0)
b_top = pd.DataFrame({'w': cv.get_feature_names_out(), 'c': b_sum}).sort_values(by='c', ascending=False).head(20)

tv = TfidfVectorizer(stop_words='english', max_features=2000)
tfidf = tv.fit_transform(df['txt'])
t_sum = tfidf.toarray().sum(axis=0)
t_top = pd.DataFrame({'w': tv.get_feature_names_out(), 's': t_sum}).sort_values(by='s', ascending=False).head(20)

print("Top 20 BoW words:\n", b_top)
print("\nTop 20 TF-IDF words:\n", t_top)
