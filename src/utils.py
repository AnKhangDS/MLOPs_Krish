import os
import sys
from dataclasses import dataclass

import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src import exception


@dataclass
class DataTransformationConfig:
    preprossor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

def save_object(file_path, obj):

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)

    except Exception as e:
        raise exception.CustomException(e, sys)
    


def evaluate_model(X_train, y_train, X_test, y_test, models:dict):

    report = {}

    for i in range(len(list(models))):
        model = list(models.values())[i]
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_model_score = r2_score(y_train, y_train_pred)
        test_model_score = r2_score(y_test, y_test_pred)

        report[list(models.keys())[i]] = test_model_score

    return report


def find_best_model(report: dict):
    # Sort by values in ascending order
    sorted_report = dict(sorted(report.items(), key=lambda item: item[1], reverse=True))
    best_model = list(sorted_report.keys())[0]
    return best_model
