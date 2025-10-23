#Image classification with modern MLP models

import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

class Patch(layers.Layer):
    def __init__(self, p): super().__init__(); self.p = p
    def call(self, x):
        b = tf.shape(x)[0]
        p = tf.image.extract_patches(x,[1,self.p,self.p,1],[1,self.p,self.p,1],[1,1,1,1],"VALID")
        return tf.reshape(p, [b, -1, p.shape[-1]])

mlp = lambda x,u,d: [layers.Dropout(d)(layers.Dense(n,activation='relu')(x)) for n in u][-1]

inp = keras.Input((32,32,3))
x = Patch(4)(inp)
x0 = layers.Dense(64)(x)
x = layers.LayerNormalization()(x0)
x = layers.Dense((32//4)**2,activation='relu')(x)
x = layers.Add()([layers.Dense(64)(x), x0])
x = layers.LayerNormalization()(x)
x = mlp(x,[128,64],0.1)
x = layers.GlobalAveragePooling1D()(x)
out = layers.Dense(10,activation='softmax')(x)

model = keras.Model(inp,out)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
hist = model.fit(x_train,y_train,epochs=5,batch_size=128,validation_split=0.1,verbose=0)

print("Accuracy:", model.evaluate(x_test,y_test,verbose=0)[1])
pred = np.argmax(model.predict(x_test),1)
ConfusionMatrixDisplay(confusion_matrix(y_test,pred),display_labels=range(10)).plot(cmap='Blues')
plt.show()
print(classification_report(y_test,pred))
plt.plot(hist.history['loss']); plt.plot(hist.history['val_loss']); plt.legend(['train','val']); plt.show()
