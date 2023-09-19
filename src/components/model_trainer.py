import os
from dataclasses import dataclass
from typing import Dict

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting data into training and test sets")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models = {
                "random_forest": RandomForestRegressor(),
                "decision_tree": DecisionTreeRegressor(),
                "gradient_boosting": GradientBoostingRegressor(),
                "linear_regression": LinearRegression(),
                "k_neighbours_regressor": KNeighborsRegressor(),
                "xgb": XGBRFRegressor(),
                "catboost": CatBoostRegressor(verbose=False),
                "adaboost": AdaBoostRegressor(),
            }

            models_report: Dict = evaluate_models(
                X_train, y_train, X_test, y_test, models=models
            )

            # Get the model name with best score from report
            best_model_name = max(models_report, key=models_report.get)
            # Get best score
            best_score = models_report[best_model_name]
            # Get best model
            best_model = models[best_model_name]

            if best_score < 0.6:
                raise CustomException("No best model found")

            logging.info(
                f"{best_model_name} was the best model found with score of {best_score}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model,
            )
            logging.info("Saved best model")

            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)

            return score

        except Exception as e:
            raise CustomException(e)
