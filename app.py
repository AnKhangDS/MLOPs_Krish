from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline import predict_pipeline



application = Flask(__name__)

app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict-data", methods=["GET", "POST"])
def predict_data():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = predict_pipeline.CustomData(
            gender = request.form.get("gender"),
            race_ethnicity = request.form.get("race_ethnicity"),
            lunch = request.form.get("lunch"),
            parental_level_of_education = request.form.get("parental_level_of_education"),
            test_preparation_course = request.form.get("test_preparation_course"),
            writing_score = request.form.get("writing_score"),
            reading_score = request.form.get("reading_score")

        )

        df = data.get_data_as_df()

        PREDICT_PIPELINE = predict_pipeline.PredictPipeline()
        results = PREDICT_PIPELINE.predict(input_features=df)

        return render_template("home.html", results=f"{results[0]:.2f}")


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)