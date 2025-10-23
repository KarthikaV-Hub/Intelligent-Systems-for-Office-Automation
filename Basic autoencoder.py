#Implement a basic autoencoder with one hidden layer for dimensionality reduction.
#Vary the size of the latent space (e.g., 2D, 10D, 50D) and analyze its effect on
#reconstruction quality.

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

(x_tr,_),(x_te,_) = mnist.load_data()
x_tr = x_tr.reshape(len(x_tr),-1)/255.0
x_te = x_te.reshape(len(x_te),-1)/255.0

def make_ae(inp_dim, lat_dim):
    inp = layers.Input((inp_dim,))
    enc = layers.Dense(lat_dim,'relu')(inp)
    dec = layers.Dense(inp_dim,'sigmoid')(enc)
    ae = models.Model(inp,dec)
    ae.compile('adam','mse')
    return ae

for latent in [2,10,50]:
    print("Latent dim:",latent)
    ae = make_ae(784,latent)
    ae.fit(x_tr,x_tr,epochs=10,batch_size=256,validation_data=(x_te,x_te),verbose=0)
    recon = ae.predict(x_te[:100])
    mse = np.mean((recon-x_te[:100])**2)
    print("Reconstruction MSE:",mse)
    
    plt.figure(figsize=(12,3))
    for i in range(8):
        plt.subplot(2,8,i+1); plt.imshow(x_te[i].reshape(28,28),cmap='gray'); plt.axis('off')
        plt.subplot(2,8,8+i+1); plt.imshow(recon[i].reshape(28,28),cmap='gray'); plt.axis('off')
    plt.suptitle(f'Latent={latent}, MSE={mse:.5f}'); plt.show()
