import os 
import sys

from src import logger, exception
from src.components import data_transformation

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join("artifacts", "train.csv")
    test_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logger.logging.info("Import the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_path), exist_ok=True)
            # os.makedirs(os.path.dirname(self.ingestion_config.test_path), exist_ok=True)
            # os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.logging.info("Train test split initiated")

            train, test = train_test_split(df, test_size=0.2, random_state=42)
            train.to_csv(self.ingestion_config.train_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_path, index=False, header=True)

            logger.logging.info("Data Ingestion has been executed")

            return (
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
            )
        except Exception as e:
            raise exception.CustomException(e, sys)
            


if __name__ == "__main__":
    logger.setup_logging()
    data_ingestor = DataIngestion()
    train_path, test_path = data_ingestor.initiate_data_ingestion()

    data_tf = data_transformation.DataTransformation()
    train_arr, test_arr, preprocessor_path = data_tf.initiate_data_transformation(train_path=train_path,
                                                                                  test_path=test_path)
    
    print(train_arr)
    print("")
    print(test_arr)
    print("")
    print(preprocessor_path)