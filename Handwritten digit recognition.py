#Copy of Handwritten digit recognition with MNIST dataset

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


digits = load_digits()
X, y = digits.data / 16.0, digits.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


for i in range(5):
    plt.imshow(X_test[i].reshape(8,8), cmap="gray")
    plt.title(f"Predicted: {y_pred[i]} | True: {y_test[i]}")
    plt.axis('off'); plt.show()
     
     
