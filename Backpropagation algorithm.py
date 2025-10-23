#Build an Artificial Neural Network by implementing the Backpropagation algorithm
#and test the same using appropriate data sets. Vary the activation functions used and
#compare the results.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sig = lambda x: 1/(1+np.exp(-x))
dsig = lambda x: sig(x)*(1-sig(x))
tanhf = np.tanh
dtanh = lambda x: 1 - np.tanh(x)**2
relu = lambda x: np.maximum(0,x)
drelu = lambda x: (x>0).astype(float)

class MiniNN:
    def __init__(self,n_in,n_hid,act='tanh',lr=0.05):
        self.lr = lr
        self.W1 = np.random.randn(n_in,n_hid)*0.1
        self.b1 = np.zeros(n_hid)
        self.W2 = np.random.randn(n_hid,1)*0.1
        self.b2 = 0.
        if act=='sigmoid': self.act,self.actp=sig,dsig
        elif act=='relu': self.act,self.actp=relu,drelu
        else: self.act,self.actp=tanhf,dtanh

    def forward(self,X):
        self.z1 = X@self.W1 + self.b1
        self.a1 = self.act(self.z1)
        self.z2 = self.a1@self.W2 + self.b2
        self.a2 = sig(self.z2)
        return self.a2

    def loss(self,y_pred,y_true):
        eps=1e-9; y=y_true.reshape(-1,1)
        return -np.mean(y*np.log(y_pred+eps)+(1-y)*np.log(1-y_pred+eps))

    def backward(self,X,y):
        m=X.shape[0]; y=y.reshape(-1,1)
        dz2=self.a2-y; dW2=self.a1.T@dz2/m; db2=np.mean(dz2)
        da1=dz2@self.W2.T; dz1=da1*self.actp(self.z1)
        dW1=X.T@dz1/m; db1=np.mean(dz1,0)
        self.W2-=self.lr*dW2; self.b2-=self.lr*db2
        self.W1-=self.lr*dW1; self.b1-=self.lr*db1

    def fit(self,X,y,epochs=500):
        losses=[]
        for e in range(epochs):
            y_pred=self.forward(X)
            losses.append(self.loss(y_pred,y))
            self.backward(X,y)
        return losses

    def predict(self,X):
        return (self.forward(X)>=0.5).astype(int).flatten()

X,y = make_moons(500,noise=0.2,random_state=0)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=1)
sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)

plt.figure(figsize=(8,5))
for act in ['tanh','sigmoid','relu']:
    net=MiniNN(2,8,act,0.05)
    l=net.fit(Xtr,ytr,epochs=500)
    ypred=net.predict(Xte)
    print(act,"Test acc:",(ypred==yte).mean())
    plt.plot(l,label=act)
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("ANN with Different Activations"); plt.legend(); plt.show()
