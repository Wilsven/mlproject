import os

import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(object, f)

    except Exception as e:
        raise CustomException(e)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    params_grid,
    cv=3,
    n_jobs=-1,
    refit=False,
    verbose=True,
):
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
