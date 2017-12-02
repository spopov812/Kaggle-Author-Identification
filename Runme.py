import sys
from Preprocessing import *
from Models import *

vocab_size = 0

if "prepare" in sys.argv:
    vocab_size = organize_data()

if "train" in sys.argv:

    if "load" in sys.argv:
        train_models(True)

    else:
        train_models()

if "predict" in sys.argv:
    predict()

print("End")