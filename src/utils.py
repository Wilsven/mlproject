import os

import dill
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(object, f)

    except Exception as e:
        raise CustomException(e)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        models_report = {}

        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            # Predict on `X_train` data
            y_train_pred = model.predict(X_train)
            # Predict on `X_test` data
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            models_report[model_name] = test_score

        return models_report

    except Exception as e:
        raise CustomException(e)
