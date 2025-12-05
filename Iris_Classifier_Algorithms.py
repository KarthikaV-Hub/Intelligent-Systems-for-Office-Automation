import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

data = load_iris()
X = data.data
y = data.target

sc = StandardScaler()
Xn = sc.fit_transform(X)

models = {
    "LogReg": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    "NB": GaussianNB(),
    "GB": GradientBoostingClassifier()
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
acc = {}
f1 = {}

for name, mdl in models.items():
    a, f = [], []
    for tr, ts in kf.split(Xn, y):
        mdl.fit(Xn[tr], y[tr])
        p = mdl.predict(Xn[ts])
        a.append(accuracy_score(y[ts], p))
        f.append(f1_score(y[ts], p, average='macro'))
    acc[name] = np.mean(a)
    f1[name] = np.mean(f)

print("Accuracy:")
for k in acc:
    print(k, round(acc[k],4))

print("\nF1 Score:")
for k in f1:
    print(k, round(f1[k],4))
