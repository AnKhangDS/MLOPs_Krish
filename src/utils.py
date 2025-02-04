import os
import sys
from dataclasses import dataclass

import dill
import numpy as np
import pandas as pd

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