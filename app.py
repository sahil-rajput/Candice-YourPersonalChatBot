######
'''
Before starting:

1) Make a new envionment in anaconda

2) Install Pyhon 3.7 inot the environment:
    open powershell (from anaconda navigator) and write: conda install python = 3.7

3) Install necessary packages using powrshell by writing:
pip install chatterbot

pip install chatterbot==1.0.0
pip install flask
pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz



pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm

'''
######

#pip install -U pip setuptools wheel
#pip install -U spacy
#python -m spacy download en_core_web_sm







from flask import Flask, render_template, request
import chatterbot

#import chatterbot-corpus
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer


from chatterbot.trainers import ListTrainer
#import spacy
import os.path
#import en_core_web_sm
#nlp = spacy.load("en_core_web_sm")
#nlp = spacy.load("en")
#nlp = en_core_web_sm.load()
#nlp = spacy.blank("en")


app = Flask(__name__)



chatbot = ChatBot('BradBot')

# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot based on the english corpus
trainer.train("chatterbot.corpus.english")

# Get a response to an input statement

#print(chatbot.get_response("What is your name?"))

trainer = ListTrainer(chatbot)
#bot.set_trainer(ListTrainer)
#bot.train(['What is your name?', 'My name is Candice'])
#bot.train(['Who are you?', 'I am a bot' ])
#bot.train(['Who created you?', 'Tony Stark', 'Sahil Rajput', 'You?'])

trainer.train(['What is your name?', 'My name is Jason'])
trainer.train(['Who are you?', 'I am Jason' ])
trainer.train(['Who created you?', 'JB-M', 'You?'])

#bot.set_trainer(ChatterBotCorpusTrainer)
#trainer = ChatterBotCorpusTrainer(bot)
#bot.train("chatterbot.corpus.english")

#trainer.train("chatterbot.corpus.english")
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(chatbot.get_response(userText))

if __name__ == "__main__":
    app.run()
