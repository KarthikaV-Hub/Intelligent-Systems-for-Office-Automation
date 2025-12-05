import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=2, random_state=42)

data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, 21)])
data['target'] = y

print(data.head())
print(data.info())
print(data['target'].value_counts())

sns.countplot(x='target', data=data)
plt.title('Class Distribution')
plt.show()

X_scaled = StandardScaler().fit_transform(data.drop('target', axis=1))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                    random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    cr = classification_report(y_test, predictions, output_dict=True)
    
    results[name] = {"Accuracy": acc, "Confusion Matrix": cm, "Classification Report": cr}
    
    print(f"--- {name} ---")
    print(f"Accuracy: {acc}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print()

accuracy_df = pd.DataFrame({name: res['Accuracy'] for name, res in results.items()}, index=[0]).T
accuracy_df.columns = ['Accuracy']
accuracy_df.sort_values('Accuracy', ascending=False, inplace=True)

plt.figure(figsize=(10, 5))
sns.barplot(x=accuracy_df.index, y='Accuracy', data=accuracy_df)
plt.xticks(rotation=45)
plt.title('Classifier Accuracy Comparison')
plt.show()
