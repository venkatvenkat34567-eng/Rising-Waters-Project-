from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/flood_model.pkl")
scaler = joblib.load("model/transform.save")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/Predict")
def predict_page():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = pd.DataFrame([{
        "Temp": float(request.form["Temp"]),
        "Humidity": float(request.form["Humidity"]),
        "Cloud Cover": float(request.form["CloudCover"]),
        "ANNUAL": float(request.form["Annual"]),
        "Jan-Feb": float(request.form["JanFeb"]),
        "Mar-May": float(request.form["MarMay"]),
        "Jun-Sep": float(request.form["JunSep"]),
        "Oct-Dec": float(request.form["OctDec"]),
        "avgjune": float(request.form["AvgJune"]),
        "sub": float(request.form["Sub"])
    }])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    if prediction == 1:
        return render_template("chance.html")
    else:
        return render_template("no_chance.html")


@app.route("/chance")
def chance():
    return render_template("chance.html")


@app.route("/no_chance")
def no_chance():
    return render_template("no_chance.html")


if __name__ == "__main__":
    app.run(debug=True)