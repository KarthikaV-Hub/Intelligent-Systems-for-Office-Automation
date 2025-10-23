#Demonstrate the different types of autoencoders using Fashion MNIST dataset and
#any industrial dataset.

import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers, losses

(xtr, _), (xte, _) = tf.keras.datasets.fashion_mnist.load_data()
xtr, xte = xtr/255.0, xte/255.0
xtr, xte = xtr[..., None], xte[..., None]
xtr_small = xtr[:10000]

def show(model, data, name):
    pred = model.predict(data[:10])
    plt.figure(figsize=(15,3))
    for i in range(10):
        plt.subplot(2,10,i+1); plt.imshow(data[i].squeeze(),cmap='gray'); plt.axis('off')
        plt.subplot(2,10,i+11); plt.imshow(pred[i].squeeze(),cmap='gray'); plt.axis('off')
    plt.suptitle(name); plt.show()

def simAE():
    i = tf.keras.Input((28,28,1))
    x = layers.Flatten()(i)
    x = layers.Dense(64,'relu')(x)
    o = layers.Dense(784,'sigmoid')(x)
    o = layers.Reshape((28,28,1))(o)
    m = models.Model(i,o)
    m.compile('adam','mse')
    m.fit(xtr_small,xtr_small,epochs=2,batch_size=128,verbose=0)
    show(m,xte,"Simple AE")

def AEcon():
    i = tf.keras.Input((28,28,1))
    x = layers.Conv2D(32,3,activation='relu',padding='same')(i)
    x = layers.MaxPooling2D(2,padding='same')(x)
    x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x = layers.MaxPooling2D(2,padding='same')(x)
    x = layers.Conv2DTranspose(64,3,2,activation='relu',padding='same')(x)
    x = layers.Conv2DTranspose(32,3,2,activation='relu',padding='same')(x)
    o = layers.Conv2D(1,3,activation='sigmoid',padding='same')(x)
    m = models.Model(i,o)
    m.compile('adam','binary_crossentropy')
    m.fit(xtr_small,xtr_small,epochs=2,batch_size=128,verbose=0)
    show(m,xte,"AE Conv")

def AEdenoise():
    n=lambda d: np.clip(d+0.3*np.random.normal(size=d.shape),0.,1.)
    i = tf.keras.Input((28,28,1))
    x = layers.Flatten()(i)
    x = layers.Dense(64,'relu')(x)
    o = layers.Dense(784,'sigmoid')(x)
    o = layers.Reshape((28,28,1))(o)
    m = models.Model(i,o)
    m.compile('adam','mse')
    m.fit(n(xtr_small),xtr_small,epochs=2,batch_size=128,verbose=0)
    show(m,n(xte),"Denoising AE")

def AEsparse():
    i=tf.keras.Input((28,28,1))
    x=layers.Flatten()(i)
    x=layers.Dense(128,'relu',activity_regularizer=regularizers.l1(1e-5))(x)
    x=layers.Dense(64,'relu')(x)
    x=layers.Dense(128,'relu')(x)
    o=layers.Dense(784,'sigmoid')(x)
    o=layers.Reshape((28,28,1))(o)
    m=models.Model(i,o)
    m.compile('adam','mse')
    m.fit(xtr_small,xtr_small,epochs=2,batch_size=128,verbose=0)
    show(m,xte,"Sparse AE")

class Samp(layers.Layer):
    def call(self, z): zm, zv = z; e=tf.random.normal(tf.shape(zm)); return zm+tf.exp(0.5*zv)*e

def VAE():
    ld=2; i=tf.keras.Input((28,28,1))
    x=layers.Flatten()(i); h=layers.Dense(128,'relu')(x)
    zm,zv=layers.Dense(ld)(h),layers.Dense(ld)(h)
    z=Samp()([zm,zv])
    e=models.Model(i,[zm,zv,z])
    li=tf.keras.Input((ld,))
    x=layers.Dense(128,'relu')(li)
    o=layers.Dense(784,'sigmoid')(x)
    o=layers.Reshape((28,28,1))(o)
    d=models.Model(li,o)
    class VAEModel(tf.keras.Model):
        def __init__(self,e,d): super().__init__(); self.e,self.d=e,d
        def call(self,x):
            zm,zv,z=self.e(x); r=self.d(z)
            rl=tf.reduce_mean(losses.binary_crossentropy(x,r))*784
            kl=-0.5*tf.reduce_mean(1+zv-tf.square(zm)-tf.exp(zv))
            self.add_loss(rl+kl); return r
    v=VAEModel(e,d)
    v.compile('adam')
    v.fit(xtr_small,xtr_small,epochs=2,batch_size=128,verbose=0)
    show(v,xte,"Variational AE")

def AEindustrial():
    n,a=np.random.normal(0,1,(10000,10)),np.random.normal(5,1,(1000,10))
    d=np.vstack([n,a]).astype('float32')
    i=tf.keras.Input((10,))
    x=layers.Dense(32,'relu')(i); x=layers.Dense(16,'relu')(x)
    x=layers.Dense(32,'relu')(x); o=layers.Dense(10)(x)
    m=models.Model(i,o)
    m.compile('adam','mse')
    m.fit(d,d,epochs=2,batch_size=128,verbose=0)
    r=m.predict(d); mse=np.mean((d-r)**2,1); t=np.percentile(mse,95)
    print(f"Anomalies detected: {(mse>t).sum()} / {len(d)}")
    plt.hist(mse,50); plt.axvline(t,color='r'); plt.title("Industrial AE"); plt.show()

simAE(); AEcon(); AEdenoise(); AEsparse(); VAE(); AEindustrial()

