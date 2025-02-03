import os 
import sys
from src import logger, exception
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    X_train_path: str = os.path.join("artifacts", "X_train.csv")
    y_train_path: str = os.path.join("artifacts", "y_train.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logger.logging.info("Import the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.X_train_path), exist_ok=True)
            # os.makedirs(os.path.dirname(self.ingestion_config.y_train_path), exist_ok=True)
            # os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.logging.info("Train test split initiated")

            X_train, y_train = train_test_split(df, test_size=0.2, random_state=42)
            X_train.to_csv(self.ingestion_config.X_train_path, index=False, header=True)
            y_train.to_csv(self.ingestion_config.y_train_path, index=False, header=True)

            logger.logging.info("Data Ingestion has been executed")

            return (
                self.ingestion_config.X_train_path,
                self.ingestion_config.y_train_path
            )
        except Exception as e:
            raise exception.CustomException(e, sys)
            


if __name__ == "__main__":
    logger.setup_logging()
    data_ingestor = DataIngestion()
    data_ingestor.initiate_data_ingestion()