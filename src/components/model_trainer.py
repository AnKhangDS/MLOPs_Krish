import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src import utils, exception, logger
from src.components import data_ingestion, data_transformation



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.logging.info("split training and test input and output data")

            X_train, y_train = (train_array[:, :-1],
                                train_array[:, -1])
            
            X_test, y_test = (test_array[:, :-1],
                              test_array[:, -1])

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }
            
            logger.logging.info("Running evalute model function to find the best learning model")

            model_report: dict = utils.evaluate_model(X_train=X_train, y_train=y_train, 
                                                X_test=X_test, y_test=y_test, models=models)

            best_model = utils.find_best_model(model_report)
            logger.logging.info(f"{best_model} is the best learning model estimated")

            model = models[best_model]

            utils.save_object(self.model_trainer_config.trained_model_file_path, model)
            logger.logging.info(f"Successfully saved the learning model")

            return best_model

        except Exception as e:
            logger.logging.info(exception.CustomException(e,sys))
            raise exception.CustomException(e, sys)
        

if __name__ == "__main__":
    logger.setup_logging()
    data_ingestor = data_ingestion.DataIngestion()
    train_path, test_path = data_ingestor.initiate_data_ingestion()

    data_tf = data_transformation.DataTransformation()
    train_arr, test_arr, preprocessor_path = data_tf.initiate_data_transformation(train_path=train_path,
                                                                                  test_path=test_path)

    model_trainer = ModelTrainer()
    model = model_trainer.initiate_model_trainer(train_array=train_arr,
                                                 test_array=test_arr)
    
    print(f"{model} Model being selected!!")


# python src/components/model_trainer.py