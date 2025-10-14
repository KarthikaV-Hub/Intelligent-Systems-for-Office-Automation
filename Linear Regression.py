#Implement Linear Regression using NumPy


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()
x = data.data[:, 3]
y = data.data[:, 2]

X = np.c_[np.ones(len(x)), x]
theta = np.linalg.inv(X.T @ X) @ X.T @ y
b, m = theta[0], theta[1]

print(f"Line: y = {m:.2f}x + {b:.2f}")

y_hat = m * x + b
plt.scatter(x, y, color="blue", label="Actual")
plt.plot(x, y_hat, color="red", label="Line")
plt.xlabel("Petal Width")
plt.ylabel("Petal Length")
plt.title("Linear Regression (Iris, NumPy)")
plt.legend()
plt.show()
     
