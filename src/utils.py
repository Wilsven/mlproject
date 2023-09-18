import os

import dill

from src.exception import CustomException


def save_preprocessor(file_path, preprocessor):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(preprocessor, f)

    except Exception as e:
        raise CustomException(e)
