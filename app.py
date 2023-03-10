from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response
from voicex import listen

import pyttsx3
#from engText import listen

app = Flask(__name__,template_folder='templates')

@app.get("/")
def index_get():
    return render_template('base.html')

@app.get("/menu")
def menu_get():
    return render_template('menu/menu.html')

@app.post("/predict")
def predict():
    text = request.get_json().get("message")

    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

@app.post("/voice")
def predicts():
        audio = request.get_json().get("message")

        # TODO: check if audio is valid
        a = listen(audio)
        response = get_response(a)

        message = {"answer": response}
        return jsonify(message)

if __name__ =="__main__":
    app.run(debug=True)
