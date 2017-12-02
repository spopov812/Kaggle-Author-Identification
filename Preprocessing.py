from Models import *
import pandas as pd
import numpy as np
import nltk

def organize_data():

    # loading data
    raw_train = pd.read_csv("train.csv")
    raw_test = pd.read_csv("test.csv")

    # separating data
    x_data = raw_train["text"]
    y_data = raw_train["author"]

    x_test = raw_test["text"]

    x_data = x_data.tolist()
    x_test = x_test.tolist()

    y_final = []

    # converting author names to a one-hot vector
    for author in y_data:
        if author == "EAP":
            y_final.append([1, 0, 0])
        elif author == "HPL":
            y_final.append([0, 1, 0])
        else:
            y_final.append([0, 0, 1])

    illegal_chars = ["\"", '.', ',', ':', ';', '?']

    # getting rid of punctuation
    for character in illegal_chars:
        x_data = [s.replace(character, "") for s in x_data]

    for character in illegal_chars:
        x_test = [s.replace(character, "") for s in x_test]

    x_verbs, x_adj, x_nouns, x_avb = [], [], [], []

    all_words = []

    # narrowing down vocab size
    for sentence in x_data:

        for word in sentence.split():
            all_words.append(word)

    for sentence in x_test:

        for word in sentence.split():
            all_words.append(word)

    word_set = set(all_words)

    # dictionary creation
    word_idx = {w: i + 1 for i, w in enumerate(word_set)}

    # print(word_idx.values())

    print("\n\nThere are %d unique words in the dataset.\n\n" % len(word_idx.items()))

    # print(word_idx.items())

    np.save("dictionary.npy", word_idx)

    # creating separate x data sets for different parts of speech
    for i, sentence in enumerate(x_data):

        if i % 1000 == 0:
            print(i, "/", len(x_data))

        # initializing sample with 0s for all words
        x_verbs.append(np.zeros(75))
        x_adj.append(np.zeros(75))
        x_nouns.append(np.zeros(75))
        x_avb.append(np.zeros(75))

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


    np.save("Adverbs.npy", x_avb)
    np.save("Nouns.npy", x_nouns)
    np.save("Adjs.npy", x_adj)
    np.save("Verbs.npy", x_verbs)

    np.save("YTrain", y_final)

    return len(word_idx.items())
