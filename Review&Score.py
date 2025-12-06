import os, re, io, random
import numpy as np
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
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

data = {
    "reviewText":[
        "Not much to write about here, but it does exac...",
        "The product does exactly as it should and is q...",
        "The primary job of this device is to block the...",
        "Nice windscreen protects my MXL mic and preven...",
        "This pop filter is great. It looks and perform...",
        "So good that I bought another one. Love the h...",
        "I have used monster cables for years, and with...",
        "I now use this cable to run from the output of...",
        "Perfect for my Epiphone Sheraton II. Monster ...",
        "Monster makes the best cables and a lifetime w..."
    ],
    "Overall":[5,4,3,5,4,4,5,5,3,4]
}
df_amz = pd.DataFrame(data)

def simple_clean(s):
    s = str(s).lower()
    s = re.sub(r'[^a-z\s]', ' ', s)
    s = re.sub(r'\s+',' ', s).strip()
    return s

df_amz['clean'] = df_amz['reviewText'].apply(simple_clean)

tfidf_vec = TfidfVectorizer(ngram_range=(1,2), max_features=200)
X = tfidf_vec.fit_transform(df_amz['clean'])
y_cls = df_amz['Overall']
y_reg = df_amz['Overall'].astype(float)

X_train, X_test, ytrain_cls, ytest_cls = train_test_split(X, y_cls, test_size=0.3, random_state=42, stratify=y_cls)
Xtr_reg, Xte_reg, ytr_reg, yte_reg = train_test_split(X.toarray(), y_reg.values, test_size=0.3, random_state=42)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ytrain_cls_encoded = le.fit_transform(ytrain_cls)
ytest_cls_encoded = le.transform(ytest_cls)

classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'NaiveBayes': MultinomialNB(),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'DecisionTree': DecisionTreeClassifier(random_state=0),
    'RandomForest': RandomForestClassifier(n_estimators=50, random_state=0),
    'GradientBoosting': GradientBoostingClassifier(random_state=0),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=0)
}

for name, clf in classifiers.items():
    try:
        if name == 'XGBoost':
            clf.fit(X_train, ytrain_cls_encoded)
            preds = clf.predict(X_test)
            preds_decoded = le.inverse_transform(preds)
            target = ytest_cls
        else:
            clf.fit(X_train, ytrain_cls)
            preds = clf.predict(X_test)
            target = ytest_cls

        print(name)
        print("Accuracy:", accuracy_score(target, preds_decoded if name=='XGBoost' else preds))
        print("Precision:", precision_score(target, preds_decoded if name=='XGBoost' else preds, average='macro', zero_division=0))
        print("Recall:", recall_score(target, preds_decoded if name=='XGBoost' else preds, average='macro', zero_division=0))
        print("F1:", f1_score(target, preds_decoded if name=='XGBoost' else preds, average='macro', zero_division=0))
        print(confusion_matrix(target, preds_decoded if name=='XGBoost' else preds))
        print(classification_report(target, preds_decoded if name=='XGBoost' else preds, zero_division=0))
    except Exception as e:
        print(name, "failed:", e)

regressors = {
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=0),
    'RandomForestRegressor': RandomForestRegressor(n_estimators=50, random_state=0),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=0),
    'XGBRegressor': XGBRegressor(objective='reg:squarederror', random_state=0)
}

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

for name, reg in regressors.items():
    try:
        reg.fit(Xtr_reg, ytr_reg)
        pred_r = reg.predict(Xte_reg)
        mae = mean_absolute_error(yte_reg, pred_r)
        rmse = np.sqrt(mean_squared_error(yte_reg, pred_r))
        r2 = r2_score(yte_reg, pred_r)
        print(name, mae, rmse, r2)
    except Exception as e:
        print(name, "failed:", e)

from sklearn.preprocessing import label_binarize
classes = sorted(df_amz['Overall'].unique())

if len(classes) > 2:
    y_test_bin = label_binarize(ytest_cls, classes=classes)
    for name, clf_model in [
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=0)),
        ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=0))
    ]:
        try:
            if name == 'XGBoost':
                clf_model.fit(X_train, ytrain_cls_encoded)
                probas = clf_model.predict_proba(X_test)
            else:
                clf_model.fit(X_train, ytrain_cls)
                probas = clf_model.predict_proba(X_test)
            from sklearn.metrics import roc_auc_score
            auc_score = roc_auc_score(y_test_bin, probas, average='macro', multi_class='ovr')
            print(name, auc_score)
        except Exception as e:
            print(name, "ROC failed:", e)

best_clf = LogisticRegression(max_iter=1000, random_state=0).fit(X_train, ytrain_cls)
preds = best_clf.predict(X_test)
cm = confusion_matrix(ytest_cls, preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion matrix")
plt.xlabel("pred")
plt.ylabel("true")
plt.show()
