# source https://keras.io/
# credits : https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras
import random
from keras.optimizers import RMSprop

class History(keras.callbacks.Callback):
    def __init__(self, filename):
        self.filename = filename
   
    def on_epoch_end(self, epoch, logs={}):
        print(epoch)
        print(logs)
        print(self.filename)

        f = open(self.filename,'a+')
        f.write("Epoch: " + str(epoch) + " ")
        f.write(str(logs))
        f.close()	

class Lstm:
    def __init__(self, text, filename=None, ephocs=0):
        self.text = text
        self.filename = filename
        self.ephocs = ephocs
        self.divide_text()

    def sample(self,preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def divide_text(self):
        chars = sorted(list(set(self.text)))
        char_indices = dict((c,i) for i,c in enumerate(chars))
        indices_char = dict((i,c) for i, c in enumerate(chars))
        
        maxlen = 40
        step = 3
        sentences = []
        next_chars = []

        for i in range(0, len(self.text) - maxlen, step):
            sentences.append(self.text[i : i + maxlen])
            next_chars.append(self.text[i+maxlen])
        
        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool) 
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool) 

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1
        
        model = Sequential()
        model.add(LSTM(128, input_shape=(maxlen, len(chars))))
        model.add(Dense(len(chars)))
        model.add(Activation('softmax'))
        
       
        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        

        #for iteration in range(1, 60):
        #    print()
        #    print('-' * 50)
        #    print('Iteration', iteration)
        #    model.fit(x, y, batch_size=128, epochs=1)
        # checkpoint
        #filepath="weights.hdf5"
        #checkpoint = ModelCheckpoint(filepath, verbose=1)
        #callbacks_list = [checkpoint]
        history = History(self.filename)
        model.fit(x, y, batch_size=128, epochs=int(self.ephocs), callbacks=[history])

        f = open(self.filename,'a+')
        start_index = random.randint(0, len(self.text) - maxlen - 1)

        generated = ''
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            generated = ''
            sentence = self.text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            print(generated)
        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

        print(generated)
        f.write(generated + "\n\n")
        f.close()	
