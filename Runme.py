import sys
from Preprocessing import *
from Models import *

if "prepare" in sys.argv:
    vocab_size = organize_data()

if "train" in sys.argv:
    train_models(vocab_size)

if "predict" in sys.argv:
    print("no")
    # TODO

print("End")