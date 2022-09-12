import pandas as pd

from src.model.titanic_model import TitanicModel
from src.model.template.base import TemplateBase


class ProjectModel(TemplateBase):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()
        self.titanic_model = TitanicModel(**kwargs)

    def fit(
            self,
            df_train: pd.DataFrame,
            df_validate: pd.DataFrame,
            df_test: pd.DataFrame,
    ):

        self.titanic_model.fit(
            df_train=df_train,
            df_validate=df_validate,
            df_test=df_test
        )

    def save_model(
            self,
            directory
    ):
        self.titanic_model.save(directory + '/titanic_model')

    def delete_model(
            self
    ):
        del self.titanic_model

    def load_model(
            self,
            directory
    ):
        self.titanic_model = \
            TitanicModel.load(directory + '/titanic_model')

    def predict(self, df: pd.DataFrame):
        df = self.titanic_model.predict(df)
        return df
