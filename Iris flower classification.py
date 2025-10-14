#Iris flower classification

#Trained a Logistic Regression model to classify iris flowers into Setosa, Versicolor, or Virginica using sepal and petal measurements.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))


sample_flower = [[6.3, 3.3, 6.0, 2.5]]  # Virginica example
prediction = model.predict(sample_flower)

print("Predicted Species:", iris.target_names[prediction[0]])

sample_flower = [[5.1, 3.5, 1.4, 0.2]]  # known Setosa example
prediction = model.predict(sample_flower)

print("Predicted Species:", iris.target_names[prediction[0])

sample_flower = [[6.0, 2.9, 4.5, 1.5]]  # Versicolor example
prediction = model.predict(sample_flower)

print("Predicted Species:", iris.target_names[prediction[0]])
     

