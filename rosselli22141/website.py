
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
app.config["DEBUG"] = True

comments = []

@app.route("/", methods=["GET", "POST"])
# @app.route("/<age><weight>", methods=["GET","POST"])
def index():

    return render_template("main_page.html")
    print(request.form["age"])



@app.route("/response", methods=["POST"])
def response():
    age = request.form["age"]
    weight = request.form["weight"]
    x = pd.DataFrame([[age, weight]], columns=["Age", "Weight"])
    clf = joblib.load("regr.pkl")
    predictions = clf.predict(x)[0]
    return render_template("main_page.html",prediction=predictions)


def train():
    df = pd.read_csv("SBP.csv")
    # x = df[["Age", "Weight"]]
    x = df[["Age", "Weight"]]
    y = df["SBP"]

    regr = LinearRegression()
    regr.fit(x, y)

    joblib.dump(regr, "regr.pkl")


def load():
    clf = joblib.load("regr.pkl")
    age = 18
    weight = 60
    x = pd.DataFrame([[age, weight]], columns=["Age", "Weight"])
    prediction = clf.predict(x)[0]
    print(prediction)

if __name__ == "__main__":
    train()
    load()

