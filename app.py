from flask import Flask, render_template, request
import pickle
import numpy as np

# load model
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():

    price = ""

    if request.method == "POST":

        age = float(request.form["age"])
        kms = float(request.form["kms"])
        brand = request.form["brand"]

        # one hot encoding manually
        maruti = 0
        tata = 0
        mahindra = 0

        if brand == "Maruti":
            maruti = 1
        elif brand == "Tata":
            tata = 1
        else:
            mahindra = 1

        data = [[age, kms, maruti, tata, mahindra]]

        prediction = model.predict(data)

        price = round(prediction[0],2)

    return render_template("home.html", price=price)


if __name__ == "__main__":
    app.run(debug=True)