#Design a single unit perceptron for classification of a linearly separable binary
#dataset.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class SimplePerceptron:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr, self.epochs = lr, epochs
    def fit(self,X,Y):
        n,f = X.shape
        self.weights = np.zeros(f); self.bias = 0
        for _ in range(self.epochs):
            errs=0
            for xi, y in zip(X,Y):
                upd = self.lr * (y - self.predict(xi))
                if upd!=0: self.weights+=upd*xi; self.bias+=upd; errs+=1
            if errs==0: break
        return self
    def net(self,X): return np.dot(X,self.weights)+self.bias
    def predict(self,X):
        if X.ndim==1: return 1 if self.net(X)>=0 else 0
        return np.where(self.net(X)>=0,1,0)

data, labels = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=0.6, random_state=42)

model = SimplePerceptron(lr=0.01, epochs=1000)
model.fit(data, labels)
print("Weights:", model.weights, "Bias:", model.bias)
plt.scatter(data[:,0], data[:,1], c=labels, cmap='coolwarm', edgecolor='k', s=60)
x_vals = np.linspace(data[:,0].min()-1, data[:,0].max()+1, 200)
if model.weights[1]!=0:
    y_vals = -(model.weights[0]*x_vals + model.bias)/model.weights[1]
    plt.plot(x_vals, y_vals, 'w-', lw=2)
plt.title("Perceptron Decision Boundary", fontsize=14)
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.grid(alpha=0.3)
plt.show()

