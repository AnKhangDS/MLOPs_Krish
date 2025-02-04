import sys
import pandas as pd
from src import exception, utils, logger

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, input_features: pd.DataFrame):

        try:

            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = utils.load_model(model_file=model_path)
            preprocessor = utils.load_preprocessor(preprocessor_file=preprocessor_path)

            # Scaled input features
            input_features_arr = preprocessor.transform(input_features)

            pred = model.predict(input_features_arr)

            return pred

        except Exception as e:
            logger.logging.info(exception.CustomException(e, sys))
            raise exception.CustomException(e, sys)


class CustomData:
    def __init__(self, 
                 gender: str,
                 race_ethnicity: str,
                 lunch: str,
                 parental_level_of_education: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.lunch = lunch
        self.parental_level_of_education = parental_level_of_education
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score


    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender], 
                "race_ethnicity": [self.race_ethnicity], 
                "lunch": [self.lunch],
                "parental_level_of_education": [self.parental_level_of_education],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]   
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            logger.logging.info(exception.CustomException(e, sys))
            raise exception.CustomException(e, sys)