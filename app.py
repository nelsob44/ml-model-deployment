import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
import pickle
load_dotenv()

tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

app = Flask(__name__)

env_config = os.getenv("PROD_APP_SETTINGS", "config.DevelopmentConfig")
app.config.from_object(env_config)

@app.route("/")

def home():
    
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get('email-content')
    tokenized_email = tokenizer.transform([email_text])
    predictions = model.predict(tokenized_email)
    prediction = 1 if predictions == 1 else -1
    return render_template('index.html', prediction=prediction, email_text=email_text)
if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000)