#Demonstrate autoencoder for reconstructing CIFAR-10 data.

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

(x_tr,_),(x_te,_) = tf.keras.datasets.cifar10.load_data()
x_tr,x_te = x_tr[:10000]/255.0, x_te/255.0

auto = models.Sequential([
    layers.Input((32,32,3)),
    layers.Conv2D(32,3,activation='relu',padding='same'),
    layers.MaxPooling2D(2,padding='same'),
    layers.Conv2D(64,3,activation='relu',padding='same'),
    layers.MaxPooling2D(2,padding='same'),
    layers.Conv2DTranspose(64,3,strides=(2,2),activation='relu',padding='same'),
    layers.Conv2DTranspose(32,3,strides=(2,2),activation='relu',padding='same'),
    layers.Conv2D(3,3,activation='sigmoid',padding='same')
])

auto.compile('adam','mse')
auto.fit(x_tr,x_tr,epochs=2,batch_size=128,validation_split=0.1)

recon = auto.predict(x_te[:10])
plt.figure(figsize=(20,4))
for i,img in enumerate(x_te[:10]):
    plt.subplot(2,10,i+1); plt.imshow(img); plt.axis('off')
    plt.subplot(2,10,i+11); plt.imshow(recon[i]); plt.axis('off')
plt.show()
