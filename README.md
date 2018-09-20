# Kaggle-Author-Identification
Predicting a text's author using Recurrent Neural Networks (LSTM modules)

# Info
Using a Kaggle dataset, this project takes .csv data, cleans and processes it and does a train/test split. Data is further processed using NLTK to tag each word in a sentence with a part of speech (POS). The POS data is collected and fed into four recurrent neural networks, each of them specializing on a single POS. After each network makes its predictions as to the author, the results are further processed and are fed into a Feed Forward Neural Network which makes the final author prediction. 90% Test set accuracy.

# Run
python Runme.py [prepare] [train] [predict]

### Prepare-
Cleans data

### Train-
Trains all models on cleaned training data

### Predict-
Uses trained models to generate predictions
