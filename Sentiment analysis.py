#Implement RNN for sentiment analysis on movie reviews 

import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

(x,y),(xt,yt)=keras.datasets.imdb.load_data(num_words=10000)
x,xt=keras.utils.pad_sequences(x,200),keras.utils.pad_sequences(xt,200)

k=keras.Sequential([
    keras.layers.Embedding(10000,128),
    keras.layers.SimpleRNN(128),
    keras.layers.Dense(1,activation='sigmoid')
])
k.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
h=k.fit(x,y,batch_size=64,epochs=5,validation_split=0.2,verbose=0)

print("Test Acc:",k.evaluate(xt,yt,verbose=0)[1])
p=(k.predict(xt)>0.5).astype(int).ravel()
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(yt,p),
                       display_labels=['Neg','Pos']).plot(cmap='Blues')
plt.show()
print(classification_report(yt,p,target_names=['Neg','Pos']))

plt.plot(h.history['accuracy']);plt.plot(h.history['val_accuracy']);plt.legend(['train','val']);plt.show()

idx={v:k for k,v in keras.datasets.imdb.get_word_index().items()}
decode=lambda s:' '.join(idx.get(i-3,'?') for i in s if i>=3)
r=xt[0];prob=k.predict(np.array([r]))[0][0]
print("\nReview:",decode(r))
print(f"T:{'Pos' if y[0] else 'Neg'} | Pred:{'Pos' if prob>0.5 else 'Neg'} ({prob:.2f})")
