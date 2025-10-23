#Implement a simple character-level RNN/LSTM for text generation (e.g., generating
#Shakespearean text).

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def load_shakes():
    path = tf.keras.utils.get_file('shakespeare.txt',
        'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
    return open(path,'r',encoding='utf-8').read()[:100000]

def prepare_data(txt, seq_len=60, step=3):
    chars = sorted(list(set(txt)))
    c2i = {c:i for i,c in enumerate(chars)}
    i2c = {i:c for c,i in c2i.items()}
    seqs, nextc = [],[]
    for i in range(0,len(txt)-seq_len,step):
        seqs.append(txt[i:i+seq_len])
        nextc.append(txt[i+seq_len])
    X = np.zeros((len(seqs),seq_len,len(chars)),dtype=np.bool_)
    y = np.zeros((len(seqs),len(chars)),dtype=np.bool_)
    for i,s in enumerate(seqs):
        for t,c in enumerate(s): X[i,t,c2i[c]]=1
        y[i,c2i[nextc[i]]]=1
    return X,y,c2i,i2c

def build_lstm(seq_len,nc):
    m = models.Sequential([
        layers.Input(shape=(seq_len,nc)),
        layers.LSTM(128),
        layers.Dense(nc,activation='softmax')
    ])
    m.compile(optimizer='adam',loss='categorical_crossentropy')
    return m

def sample_next(preds,temp=0.7):
    p = np.log(np.asarray(preds,dtype='float64')+1e-9)/temp
    p = np.exp(p)/np.sum(np.exp(p))
    return np.argmax(np.random.multinomial(1,p,1))

txt = load_shakes()
X,y,c2i,i2c = prepare_data(txt)
model = build_lstm(60,len(c2i))
model.fit(X,y,batch_size=128,epochs=5,verbose=0)

idx = np.random.randint(0,len(txt)-60)
seed = txt[idx:idx+60]
gen = seed

for _ in range(400):
    x_pred = np.zeros((1,60,len(c2i)))
    for t,ch in enumerate(gen[-60:]):
        if ch in c2i: x_pred[0,t,c2i[ch]]=1
    preds = model.predict(x_pred,verbose=0)[0]
    gen += i2c[sample_next(preds,0.7)]

print("\nGenerated Text:\n")
print(gen)

