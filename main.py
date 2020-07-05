# A program that loads data by extracting from json file, represent each sentence with a list of words
# in our models vocabulary, where Each position in the list will represent a word from our vocabulary.
# If the position in the list is a 1 then that will mean that the word exists in our sentence, and
# if it is a 0 then the word is not present.
#
# Then we develop a model, with neural network with two hidden layers. The goal of our network will be
# to look at a bag of words and give a class that they belong to (one of our tags from the json file).
# We use tensorflow tflearn to build and connect our neural network.
#
# Then we do training and saving the model that we developed. We "fit" our data to the model, and save it.
# We can load the model without training again if we already processed and saved.
#
# Then we finally use the model. We want to generate a response to any sentence the user types in. Again,
# we take the input as a bag of words, and it generates a list of probabilities for all of our tags.

# – Get some input from the user
# – Convert it to a bag of words
# – Get a prediction from the model
# – Find the most probable class
# – Pick a response from that class


import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents/seonjun.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    list_pattern = []
    list_tag = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:  # for each patter in from each intent
            tokens = nltk.word_tokenize(pattern)  # turn pattern into a list of words using tokenize
            words.extend(tokens)
            list_pattern.append(tokens)  # add each list of words to patter_x
            list_tag.append(intent["tag"])  # add the associated tag to tag_y

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    # one hot encoded
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(list_pattern):
        bag = []

        stems = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in stems:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(list_tag[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

# take input as a bag of words and outputs some kinds of labels telling we should respond with tags
net = tflearn.input_data(shape=[None, len(training[0])])  # input layer data (length of training data)
net = tflearn.fully_connected(net, 8)  # hidden layer with 8 neurons
net = tflearn.fully_connected(net, 8)  # hidden layer with 8 neurons, so two hidden layer

# connect to output layer with softmax activation function that gives probability.
# Neurons represent each of our classes, so each neuron is tag, like hello, goodbye, etc.
# So our model predicts which tag we should take responsible to give to user.
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# DNN = just type of neural network
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")  # only add this line and try/except when we already produced the model.
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)  # index of the greatest number (probability)
        tag = labels[results_index]  # the tag (greeting, hours, etc.)

        if results[results_index] > 0.5:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I didn't get that, try again.")


chat()
