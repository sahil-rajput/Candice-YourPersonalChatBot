######
'''
Using this useful guide: https://stackoverflow.com/questions/56268480/can-not-install-chatterbot-in-anaconda

Before starting:

1) Make a new envionment in anaconda

2) Install Pyhon 3.7 inot the environment:
    open powershell (from anaconda navigator) and write: conda install python = 3.7

3) Install necessary packages using powrshell by writing:


pip install chatterbot==1.0.0
pip install chatterbot-corpus==1.2.0
pip install flask

Not sure any of this is needed
pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm

'''
######

from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
import os.path


app = Flask(__name__)

chatbot = ChatBot('BradBot')

# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot based on the english corpus # can make a .yml file
trainer.train("chatterbot.corpus.english")

# Get a response to an input statement
#print(chatbot.get_response("What is your name?"))

trainer = ListTrainer(chatbot)

trainer.train(['What is your name?', 'My name is Jason'])
trainer.train(['Who are you?', 'I am Jason'])
trainer.train(['Who created you?', 'D&I', 'You?'])


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(chatbot.get_response(userText))


if __name__ == "__main__":
    app.run(debug=True)
