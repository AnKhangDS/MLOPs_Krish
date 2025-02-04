import os
import sys
from dataclasses import dataclass

import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


from src import exception, logger


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
    


def evaluate_model(X_train, y_train, X_test, y_test, models:dict, params:dict):
    report = {}

    for model_name, model in models.items():
        para = params.get(model_name, {})  # Get the corresponding hyperparameters

        # Perform GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(estimator=model, param_grid=para, cv=3)
        gs.fit(X_train, y_train)

        # Update model with the best parameters found by GridSearchCV
        model.set_params(**gs.best_params_)

        # Fit the model with the best parameters
        model.fit(X_train, y_train)

        # Predict on both train and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate R2 scores for both train and test sets
        train_model_score = r2_score(y_train, y_train_pred)
        test_model_score = r2_score(y_test, y_test_pred)

        # Add test model score to the report dictionary
        report[model_name] = test_model_score

    return report



def find_best_model(report: dict):
    # Sort by values in ascending order
    sorted_report = dict(sorted(report.items(), key=lambda item: item[1], reverse=True))

    best_model_score = list(sorted_report.values())[0]
    if best_model_score < 0.6:
        logger.logging.info("No best model found")
        logger.logging.info(f"Best model accuracy: {best_model_score:.2f}")
        logger.logging.info("Should find better data or optimize the model again")

    best_model = list(sorted_report.keys())[0]
    logger.logging.info(f"Model accuracy: {best_model_score:.2f}")
    return best_model




