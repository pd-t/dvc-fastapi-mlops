import abc

import pandas as pd
from ludwig.api import LudwigModel

from src.model.template.base import TemplateBase


class LudwigTemplate(TemplateBase, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self.ludwig_model = None
        self.statistics = None

    def save_model(self, directory: str):
        self.statistics = None
        self.ludwig_model.save(directory + '/ludwig')

    def delete_model(self):
        del self.ludwig_model

    def load_model(self, directory: str):
        self.ludwig_model = LudwigModel.load(directory + '/ludwig')

    def fit(
            self,
            df_train: pd.DataFrame,
            df_validate: pd.DataFrame,
            df_test: pd.DataFrame,
    ):
        self.ludwig_model = self.generate()
        self.statistics = self.ludwig_model.train(
            training_set=self.preprocess(df_train),
            validation_set=self.preprocess(df_validate),
            test_set=self.preprocess(df_test)
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        predictions, dummy = self.ludwig_model.predict(self.preprocess(df))
        predictions.index = df.index
        df = pd.concat([df, predictions], axis=1)
        return df

    @abc.abstractmethod
    def generate(self) -> LudwigModel:
        pass

    @abc.abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
