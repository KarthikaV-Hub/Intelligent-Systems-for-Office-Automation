import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

n = 10000
m = 200
e = 128
b = 64
t = 5

(x1, y1), (x2, y2) = keras.datasets.imdb.load_data(num_words=n)
x1 = pad_sequences(x1, maxlen=m)
x2 = pad_sequences(x2, maxlen=m)

model = Sequential()
#Implement the Deep learning algorithm for sentiment analysis

model.add(Embedding(n, e))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x1, y1, epochs=t, batch_size=b, validation_split=0.2)

loss, acc = model.evaluate(x2, y2)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

model.save("sentiment_model.keras")

word_index = keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

tokenizer = Tokenizer(num_words=n)
tokenizer.word_index = word_index

def text_to_seq(text):
    seq = []
    for w in text.lower().split():
        seq.append(word_index.get(w, 2))  # 2 = <UNK>
    return pad_sequences([seq], maxlen=m)

def predict_sent(text):
    seq = text_to_seq(text)
    p = model.predict(seq, verbose=0)
    print("Input:", text)
    print("Predicted probability positive:", float(p[0][0]))
    print("Predicted sentiment:", "positive" if p[0][0]>=0.5 else "negative")

predict_sent("I hate this movie")
predict_sent("This is the best movie I have seen")
predict_sent("I absolutely loved this film")
