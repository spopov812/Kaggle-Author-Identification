from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Flatten, Embedding
from keras.layers import LeakyReLU
import numpy as np
from keras.callbacks import ModelCheckpoint


def POS_model(vocab_size):

    model = Sequential()

    model.add(Embedding(vocab_size, 32, input_length=75))

    model.add(LSTM(64))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(.3))

    model.add(LSTM(64))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(.3))

    model.add(LSTM(64))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(.3))

    model.add(Flatten())

    model.add(Dense(3), activation="softmax")

    model.compile(

        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=[]

    )

    return model

def final_prediction_model():

    model = Sequential()

    model.add(Dense(output=64, input_shape=(4,), activation="relu"))
    model.add(Dropout(.4))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(.4))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(.4))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(.4))

    model.add(Dense(3, activation="softmax"))

    model.compile(

        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=[]

    )

    return model


def train_models(load=False):
    x_verbs, x_adj, x_nouns, x_avb = np.load("Adverbs.npy"), np.load("Nouns.npy"), np.load("Adjs.npy"), \
                                     np.load("Verbs.npy")

    y_train = np.load("YTrain")

    verb_model, adj_model, noun_model, avb_model = POS_model(), POS_model(), POS_model(), POS_model()
    final_prediction = final_prediction_model()

    epochs = 20

    if load:
        verb_model.load_weights(input("Verb filename.\n"))
        adj_model.load_weights(input("\nAdjective filename.\n"))
        noun_model.load_weights(input("\nNoun filename.\n"))
        avb_model.load_weights(input("\nAdverb filename.\n"))

        epochs = input("How many epochs of training?")

    for i in range(epochs):
        print("\n\n\nBEGINNING EPOCH ", i + 1, "\n\n\n")

        verb_model.fit(x_verbs, y_train, epochs=1, batch_size=64, callbacks=custom_callback("Verbs"))

        adj_model.fit(x_adj, y_train, epochs=1, batch_size=64, callbacks=custom_callback("Adjs"))

        noun_model.fit(x_nouns, y_train, epochs=1, batch_size=64, callbacks=custom_callback("Nouns"))

        avb_model.fit(x_avb, y_train, epochs=1, batch_size=64, callbacks=custom_callback("Avbs"))

    predictions = []

    for i in range(len(x_verbs)):
        one_sample = []

        one_sample.append(verb_model.predict(x_verbs[i]))
        one_sample.append(adj_model.predict(x_adj[i]))
        one_sample.append(noun_model.predict(x_nouns[i]))
        one_sample.append(avb_model.predict(x_avb[i]))

        predictions.append(one_sample)

    final_prediction(predictions, y_train, epochs=(epochs * 2.5), batch_size=64, callbacks=custom_callback("FinalPred"))


def custom_callback(filename):
    callbacks = []

    name = filename + "_{epoch:02d}_{categorical_crossentropy:.2f}.h5"

    callbacks.append(ModelCheckpoint(name,
                                     monitor='categorical_crossentropy', verbose=0,
                                     save_best_only=True, save_weights_only=False, mode='auto', period=1))

