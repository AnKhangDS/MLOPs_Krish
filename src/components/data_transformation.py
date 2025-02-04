import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import exception, logger, utils 


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = utils.DataTransformationConfig()

    def get_data_transformer_object(self, target_col="math_score", filename="artifacts/train.csv"):
        try:
            df = pd.read_csv(filename)
            X = df.drop(columns=[target_col])
            num_features = X.select_dtypes(exclude="object").columns
            cat_features = X.select_dtypes(include="object").columns
            print(num_features)

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))

                ]

            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="constant")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ("scaler", StandardScaler(with_mean=False))
                ]

            )

            logger.logging.info("Categorical columns have just completely encoded and scaled")

            logger.logging.info(f"Numerical features: {num_features}")
            logger.logging.info(f"Caterical features: {cat_features}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipline", cat_pipeline ,cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            logger.logging.info(exception.CustomException(e, sys))
            raise exception.CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.logging.info("Initialize train and test dataset")
            logger.logging.info("Obtaining preprocessing objects")

            target_col = "math_score"
            preprocessing_obj = self.get_data_transformer_object(target_col=target_col, filename=train_path)

            input_features_train_df = train_df.drop(columns=[target_col], axis=1)
            target_train_df = train_df[target_col]

            input_features_test_df = test_df.drop(columns=[target_col], axis=1)
            target_test_df = test_df[target_col]

            logger.logging.info("Activating preprocessing object on train and test dataframe")

            # Automatically transform df to numpy array
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            # Concatenate input_features array with target array (2D array)
            train_arr = np.c_[input_features_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_test_df)]

            utils.save_object(
                
                file_path = self.data_transformation_config.preprossor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprossor_obj_file_path
            )


        except Exception as e:
            logger.logging.info(exception.CustomException(e, sys))
            raise exception.CustomException(e, sys)

