#Preprocessing

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

num_cols = X.select_dtypes(include="number").columns
X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
X[num_cols] = StandardScaler().fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("\nFirst 5 rows of preprocessed features:\n", pd.DataFrame(X_train, columns=X.columns).head())
     
     
