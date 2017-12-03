import sys
from Preprocessing import *
from Models import *

vocab_size = 32631

if "prepare" in sys.argv:
    vocab_size = organize_data()

if "train" in sys.argv:

    if "load" in sys.argv:
        train_models(vocab_size, True)

    else:
        train_models(vocab_size)

if "predict" in sys.argv:
    predict()


