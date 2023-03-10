import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import speech_recognition as sr
from time import ctime
import time
import os
from gtts import gTTS
import requests, json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading our main Data
with open("data.json", "r") as f:
    data = json.load(f)

# Loading saved model file
FILE = "SavedData.pth"
loadData = torch.load(FILE)

# Loading data from Saved Data dictonary to model
input_size = loadData["input_size"]
hidden_size = loadData["hidden_size"]
output_size = loadData["output_size"]
all_words = loadData["all_words"]
tags = loadData["tags"]
model_state = loadData["model_state"]

# Creating Data
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)  # Now the model knows our train parameters
model.eval()

# implementing ChatBot
bot_name = "E.D.I.T.H"


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in data['data']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."


def listen(audio):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("I am listening...")
        audio = r.listen(source, timeout=10)
    data = ""
    try:
        data = r.recognize_google(audio)
        print("You " + data)

    except sr.RequestError as e:
        print("Google Speech Recognition did not understand audio")
        print("Request Failed; {0}".format(e))

    print(data)
    resp = get_response(data)
    print(resp)

    return data


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        if sentence == "voice":
            sentence = listen()

        resp = get_response(sentence)
        print(resp)