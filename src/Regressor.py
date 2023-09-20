import os
import re
import warnings
import pandas as pd
from typing import List
from sklearn.exceptions import NotFittedError
from schema.data_schema import RegressionSchema
from autogluon.tabular import TabularDataset, TabularPredictor
from joblib import load, dump
from config import paths

warnings.filterwarnings("ignore")
PREDICTOR_FILE_NAME = "predictor.joblib"


def clean_and_ensure_unique(names: List[str]) -> List[str]:
    """
    Clean the provided column names by removing special characters and ensure their
    uniqueness.

    The function first removes any non-alphanumeric character (except underscores)
    from the names. Then, it ensures the uniqueness of the cleaned names by appending
    a counter to any duplicates.

    Args:
        names (List[str]): A list of column names to be cleaned.

    Returns:
        List[str]: A list of cleaned column names with ensured uniqueness.

    Example:
        >>> clean_and_ensure_unique(['3P%', '3P', 'Name', 'Age%', 'Age'])
        ['3P', '3P_1', 'Name', 'Age', 'Age_1']
    """

    # First, clean the names
    cleaned_names = [re.sub("[^A-Za-z0-9_]+", "", name) for name in names]

    # Now ensure uniqueness
    seen = {}
    for i, name in enumerate(cleaned_names):
        original_name = name
        counter = 1
        while name in seen:
            name = original_name + "_" + str(counter)
            counter += 1
        seen[name] = True
        cleaned_names[i] = name

    return cleaned_names


class Regressor:
    """A wrapper class for the AutoGluon Regressor.

    This class provides a consistent interface that can be used with other
    regressor models.
    """

    def __init__(self, train_input: TabularDataset, schema: RegressionSchema):
        """Construct a new Regressor."""
        self._is_trained: bool = False
        self.predictor: TabularPredictor = TabularPredictor(
            label=schema.target, eval_metric='root_mean_squared_error', path=paths.PREDICTOR_DIR_PATH
        )
        self.train_input = train_input
        self.schema = schema
        self.model_name = 'AutoGluon_regressor'

    def __str__(self):
        return f"Model name: {self.model_name}"

    def train(self) -> None:
        """Train the model on the provided data"""
        self.predictor.fit(train_data=self.train_input)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> pd.Series:
        """Predict labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.

        Returns:
            pd.Series: The output predictions.
        """
        return self.predictor.predict(inputs)

    def save(self, model_dir_path: str) -> None:
        """Save the regressor to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """

        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Regressor":
        """Load the regressor from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Regressor: A new instance of the loaded regressor.
        """
        return load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))


def predict_with_model(model: "Regressor", data: pd.DataFrame) -> pd.Series:
    """
    Predict labels for the given data.

    Args:
        model (Regressor): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        pd.Series: The predicted labels.
    """
    return model.predict(data)


def save_predictor_model(model: "Regressor", predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> "Regressor":
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)
