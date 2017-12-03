from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Flatten, Embedding
from keras.layers import LeakyReLU
import numpy as np
import pandas as pd
import nltk
from keras.callbacks import ModelCheckpoint


def pos_model(vocab_size):

    model = Sequential()

    model.add(Embedding(vocab_size, 32, input_length=300))

    model.add(LSTM(64, return_sequences=True))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(.3))

    model.add(LSTM(64, return_sequences=True))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(.3))

    model.add(LSTM(64, return_sequences=True))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(.3))

    model.add(Flatten())

    model.add(Dense(3, activation="softmax"))

    model.compile(

        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=[]

    )

    return model


def final_prediction_model():

    model = Sequential()

    model.add(Dense(64, input_shape=(4,), activation="relu"))
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


def train_models(vocab_size, load=False):

    # loading x training data
    x_verbs, x_adj, x_nouns, x_avb = np.load("Adverbs.npy"), np.load("Nouns.npy"), np.load("Adjs.npy"), \
                                     np.load("Verbs.npy")
    # loading y training data
    y_train = np.load("YTrain.npy")

    # creation of proper models
    verb_model, adj_model, noun_model, avb_model = pos_model(vocab_size), pos_model(vocab_size), \
                                                    pos_model(vocab_size), pos_model(vocab_size)

    final_prediction = final_prediction_model()

    epochs = 20

    # if user wants to load model weights
    if load:
        verb_model.load_weights(input("Verb filename.\n"))
        adj_model.load_weights(input("\nAdjective filename.\n"))
        noun_model.load_weights(input("\nNoun filename.\n"))
        avb_model.load_weights(input("\nAdverb filename.\n"))

        epochs = input("How many epochs of training?")

    # training of pos models
    for i in range(epochs):
        print("\n\n\nBEGINNING EPOCH ", i + 1, "\n\n\n")

        verb_model.fit(x_verbs, y_train, epochs=1, batch_size=64, callbacks=custom_callback("Verbs"))

        adj_model.fit(x_adj, y_train, epochs=1, batch_size=64, callbacks=custom_callback("Adjs"))

        noun_model.fit(x_nouns, y_train, epochs=1, batch_size=64, callbacks=custom_callback("Nouns"))

        avb_model.fit(x_avb, y_train, epochs=1, batch_size=64, callbacks=custom_callback("Avbs"))

    predictions = []

    # each pos model generates its predictions for each sample which is then used to train the final prediction
    # model
    for i in range(len(x_verbs)):
        one_sample = []

        one_sample.append(verb_model.predict(x_verbs[i]))
        one_sample.append(adj_model.predict(x_adj[i]))
        one_sample.append(noun_model.predict(x_nouns[i]))
        one_sample.append(avb_model.predict(x_avb[i]))

        predictions.append(one_sample)

    # training the final prediction model
    final_prediction(predictions, y_train, epochs=(epochs * 2.5), batch_size=64, callbacks=custom_callback("FinalPred"))


# creation of callback to save model training progress
def custom_callback(filename):
    callbacks = []

    name = filename + "/" + filename + "_{epoch:02d}_{categorical_crossentropy:.2f}.h5"

    callbacks.append(ModelCheckpoint(name,
                                     monitor='categorical_crossentropy', verbose=0,
                                     save_best_only=True, save_weights_only=False, mode='auto', period=1))


def predict():
    raw_test = pd.read_csv("test.csv")

    x_data_words = raw_test["text"]

    x_data_words = x_data_words.tolist()

    word_idx = np.load("dictionary.npy")

    x_verbs, x_adj, x_nouns, x_avb = [], [], [], []

    for sentence in x_data_words:

        # initializing sample with 0s for all words
        x_verbs.append(np.zeros(75))
        x_adj.append(np.zeros(75))
        x_nouns.append(np.zeros(75))
        x_avb.append(np.zeros(75))

        for i, word in enumerate(sentence.split()):

            token = nltk.pos_tag(nltk.word_tokenize(word))

            if type(token) is tuple:
                token = [token]

            if not token:
                print("Uh oh")
                continue

            # inserting proper word into proper array
            if token[0][1] == 'RB':
                x_avb[-1][i] = word_idx.get(word)

            elif token[0][1] == 'NN':
                x_nouns[-1][i] = word_idx.get(word)

            elif token[0][1] == 'JJ':
                x_adj[-1][i] = word_idx.get(word)

            elif token[0][1] == 'VBP' or token[0][1] == 'VB':
                x_verbs[-1][i] = word_idx.get(word)

    verb_model, adj_model, noun_model, avb_model = pos_model(), pos_model(), pos_model(), pos_model()
    final_prediction = final_prediction_model()

    verb_model.load_weights(input("Verb filename.\n"))
    adj_model.load_weights(input("\nAdjective filename.\n"))
    noun_model.load_weights(input("\nNoun filename.\n"))
    avb_model.load_weights(input("\nAdverb filename.\n"))
    final_prediction.load_weights(input("\nFinal model filename.\n"))

    predictions = []

    # each pos model generates its predictions for each sample which is then used in final prediction model
    for i in range(len(x_verbs)):
        one_sample = []

        one_sample.append(verb_model.predict(x_verbs[i]))
        one_sample.append(adj_model.predict(x_adj[i]))
        one_sample.append(noun_model.predict(x_nouns[i]))
        one_sample.append(avb_model.predict(x_avb[i]))

        predictions.append(one_sample)

    # final prediction
    author_predictions = final_prediction.predict(predictions)

    print(author_predictions)

    '''
    # loading testing data
    df = pd.read_csv("test.csv")
    df = df.drop(["text"])

    df.assign(author=author_predictions)

    # saving final predictions to csv
    df.to_csv("Final_Submission.csv")
    '''
