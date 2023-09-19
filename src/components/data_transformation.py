import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_preprocessor(self) -> ColumnTransformer:
        """Creates a preprocessor pipeline.

        Creates a preprocessor with two transformer pipelines for numerical and categorical features.

        Raises:
            CustomException: Raises a custom exception.

        Returns:
            ColumnTransformer: Preprocessor object.
        """
        try:
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(
                f"Numerical features: {numerical_features} standard scaling completed"
            )
            logging.info(
                f"Categorical features: {categorical_features} encoding completed"
            )

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_features),
                    (
                        "categorical_pipeline",
                        categorical_pipeline,
                        categorical_features,
                    ),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e)

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> tuple(np.ndarray, np.ndarray, str):
        """Initiates data transformation.

        Preprocesses and transforms the data. First, fits the preprocessor on the train data which
        learns the statistics and then transform the test data.

        Args:
            train_path (str): Path to training data.
            test_path (str): Path to testing data.

        Raises:
            CustomException: Raises a custom exception.

        Returns:
            tuple(np.ndarray, np.ndarray, str): Returns the preprocessed and transformed
                                                train and test arrays as well as the file path
                                                to the preprocessor.pkl file.
        """
        try:
            train_df = pd.read_csv(train_path, index_col=False)
            test_df = pd.read_csv(test_path, index_col=False)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor")
            preprocessor = self.get_data_transformer_preprocessor()

            target_column = "math_score"
            input_features_train_df = train_df.drop(target_column, axis=1)
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(target_column, axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe"
            )
            input_features_train_arr = preprocessor.fit_transform(
                input_features_train_df
            )
            input_features_test_arr = preprocessor.transform(input_features_test_df)

            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                object=preprocessor,
            )
            logging.info("Saved preprocessor")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path,
            )

        except Exception as e:
            raise CustomException(e)
