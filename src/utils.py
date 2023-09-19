import os
from typing import Any, Dict

import dill
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path: str, object: Any):
    """Saves an object.

    Saves an object to a specific file path.

    Args:
        file_path (str): File path.
        object (Any): A model or preprocessor object.

    Raises:
        CustomException: Raises a custom exception.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(object, f)

    except Exception as e:
        raise CustomException(e)


def load_object(file_path: str) -> Any:
    """Loads an object.

    Loads an object from a specific file path.

    Args:
        file_path (str): File path.

    Raises:
        CustomException: Raises a custom exception.

    Returns:
        Any: A model or preprocessor object.
    """
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)

    except Exception as e:
        raise CustomException(e)


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, Any],
    params_grid: Dict[str, Dict[str, Any]],
    cv: int = 3,
    n_jobs: int = -1,
    refit: bool = False,
    verbose: int = 1,
) -> tuple:
    """Evaluate regressor models.

    Evaludate the regressor models using grid search over all hyperparameters to get the best
    hyperparameters that give the best R-squared. These are stored in dictionaries and returned.

    Args:
        X_train (np.ndarray): X_train features array.
        y_train (np.ndarray): y_train labels array.
        X_test (np.ndarray): X_test features array.
        y_test (np.ndarray): y_test labels array.
        models (Dict[str, Any]): Dictionary of different regressor models.
        params_grid (Dict[str, Dict[str, Any]]): Dictionary of hyperparameters for tuning.
        cv (int, optional): Number of cross-validations. Defaults to 3.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
        refit (bool, optional): Refit an estimator using the best found parameters on the whole dataset. Defaults to False.
        verbose (int, optional): Controls the verbosity: the higher, the more messages. Defaults to 1.

    Raises:
        CustomException: Raises a custom exception.

    Returns:
        tuple: Tuple of dictionaries containing models' best scores and hyperparameters.
    """
    try:
        models_report = {}
        models_best_params = {}

        for model_name, model in models.items():
            # Get hyperparameters
            param_grid = params_grid[model_name]

            # Grid search for best hyperparameters
            gs = GridSearchCV(
                model, param_grid, cv=cv, n_jobs=n_jobs, refit=refit, verbose=verbose
            )
            gs.fit(X_train, y_train)

            # Set the best hyperparameters found from grid search
            model.set_params(**gs.best_params_)
            # Train model
            model.fit(X_train, y_train)
            # Predict on `X_train` data
            y_train_pred = model.predict(X_train)
            # Predict on `X_test` data
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            models_report[model_name] = test_score
            models_best_params[model_name] = gs.best_params_

        return models_report, models_best_params

    except Exception as e:
        raise CustomException(e)
