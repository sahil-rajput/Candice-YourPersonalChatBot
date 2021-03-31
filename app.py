from flask import Flask, render_template, request
import chatterbot
#import ChatterBot-corpus
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
import spacy
import os.path
#import en_core_web_sm
#nlp = spacy.load("en_core_web_sm")
#nlp = en_core_web_sm.load()

app = Flask(__name__)

bot = ChatBot("Candice")

trainer = ListTrainer(bot)
#bot.set_trainer(ListTrainer)
#bot.train(['What is your name?', 'My name is Candice'])
#bot.train(['Who are you?', 'I am a bot' ])
#bot.train(['Who created you?', 'Tony Stark', 'Sahil Rajput', 'You?'])

trainer.train(['What is your name?', 'My name is Candice'])
trainer.train(['Who are you?', 'I am a bot' ])
trainer.train(['Who created you?', 'Tony Stark', 'Sahil Rajput', 'You?'])

#bot.set_trainer(ChatterBotCorpusTrainer)
trainer = ChatterBotCorpusTrainer(bot)
#bot.train("chatterbot.corpus.english")
#trainer.train("chatterbot.corpus.english")
trainer.train("chatterbot.corpus.english")
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(bot.get_response(userText))

if __name__ == "__main__":
    app.run()
