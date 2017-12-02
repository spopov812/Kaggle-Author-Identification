import sys
from Preprocessing import *

if "prepare" in sys.argv:
    organize_data()

if "train" in sys.argv:
    train_models()

if "predict" in sys.argv:
    print("no")
    # TODO

print("End")