from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("xgb_glucose_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET","POST"])
def home():

    prediction = None

    if request.method == "POST":

        ethanol = float(request.form["ethanol"])
        heart_rate = float(request.form["heart_rate"])
        conductance = float(request.form["conductance"])
        temperature = float(request.form["temperature"])

        data = np.array([[ethanol, heart_rate, conductance, temperature]])

        scaled_data = scaler.transform(data)

        prediction = model.predict(scaled_data)[0]

        prediction = round(prediction,2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)