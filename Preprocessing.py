from Models import *
import pandas as pd
import numpy as np
import nltk


def organize_data():

    # loading data
    raw_data = pd.read_csv("train.csv")

    # separating data
    x_data = raw_data["text"]
    y_data = raw_data["author"]

    y_final = []

    # converting author names to a one-hot vector
    for author in y_data:
        if author == "":
            y_final.append([1, 0, 0])
        elif author == "":
            y_final.append([0, 1, 0])
        else:
            y_final.append([0, 0, 1])

    illegal_chars = ["\"", '.', ',', ':', ';', '?']

    # getting rid of punctuation
    for i, sentence in enumerate(x_data):

        # print(type(sentence))
        for char in illegal_chars:
            x_data[i] = sentence.replace(char, "")

    x_verbs, x_adj, x_nouns, x_avb = [], [], [], []

    all_words = []

    for sentence in x_data:

        for word in sentence.split():
            all_words.append(word)


    # dictionary creation
    word_idx = {w: i for w, i in enumerate(set(all_words))}

    print("\n\nThere are %d unique words in the dataset.\n\n" % len(word_idx.items()))

    print(word_idx.items())

    np.save("dictionary.npy", word_idx)

    # creating separate x data sets for different parts of speech
    for i, sentence in enumerate(x_data):

        if i % 1000 == 0:
            print(i, "/", len(x_data))

        # initializing sample with 0s for all words
        x_verbs.append(np.zeros(len(sentence.split())))
        x_adj.append(np.zeros(len(sentence.split())))
        x_nouns.append(np.zeros(len(sentence.split())))
        x_avb.append(np.zeros(len(sentence.split())))

        for j, word in enumerate(sentence.split()):

            token = nltk.pos_tag(nltk.word_tokenize(word))

            if type(token) is tuple:
                token = [token]

            if not token:
                print("Uh oh")
                continue

            # inserting proper word into proper array
            if token[0][1] == 'RB':
                x_avb[-1][j] = word_idx.get(word)

            elif token[0][1] == 'NN':
                x_nouns[-1][j] = word_idx.get(word)

            elif token[0][1] == 'JJ':
                x_adj[-1][j] = word_idx.get(word)

            elif token[0][1] == 'VBP' or token[0][1] == 'VB':
                x_verbs[-1][j] = word_idx.get(word)


def train_models():
    print("no")