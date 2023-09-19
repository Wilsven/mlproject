import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
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

    def initiate_model_trainer(
        self, train_arr: np.ndarray, test_arr: np.ndarray
    ) -> float:
        """Initiates model trainer.

        Initiates model training of various regressor models. Stores the R-squared scores
        as well as the best parameters used to tune each regressor model.

        Args:
            train_arr (np.ndarray): Train data array. Contains the training features and labels.
            test_arr (np.ndarray): Test data array. Contains the test features and labels.

        Raises:
            CustomException: Raises custom exception.

        Returns:
            float: The best R-squared score.
        """
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

            params_grid = {
                "random_forest": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    "max_features": ["sqrt", "log2", None],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "decision_tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    "splitter": ["best", "random"],
                    "max_features": ["sqrt", "log2"],
                },
                "gradient_boosting": {
                    "loss": ["squared_error", "huber", "absolute_error", "quantile"],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "criterion": ["squared_error", "friedman_mse"],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "linear_regression": {"fit_intercept": [True, False]},
                "k_neighbours_regressor": {
                    "n_neighbors": [5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["ball_tree", "kd_tree", "brute"],
                },
                "xgb": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "catboost": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "adaboost": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "loss": ["linear", "square", "exponential"],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            models_report, models_best_params = evaluate_models(
                X_train, y_train, X_test, y_test, models=models, params_grid=params_grid
            )

            # Get the model name with best score from report
            best_model_name = max(models_report, key=models_report.get)
            # Get best score
            best_score = models_report[best_model_name]
            # Get model's best hyperparameters
            best_params = models_best_params[best_model_name]
            # Get best model
            best_model = models[best_model_name]

            if best_score < 0.6:
                raise CustomException("No best model found")

            logging.info(
                f"{best_model_name} was the best model found with score of {best_score} with {best_params}"
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
