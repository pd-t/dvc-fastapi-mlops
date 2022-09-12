import abc
import pickle

import pandas as pd


class TemplateBase(metaclass=abc.ABCMeta):

    @classmethod
    def load(cls, directory: str):
        template = cls()
        file = open(directory + '/template.model', 'rb')
        template.__dict__ = pickle.load(file)
        file.close()
        template.load_model(directory)
        return template

    def save(self, directory: str):
        self.save_model(directory)
        self.delete_model()
        file = open(directory + '/template.model', 'wb')
        pickle.dump(self.__dict__.copy(), file, pickle.HIGHEST_PROTOCOL)
        file.close()
        self.load_model(directory)

    @abc.abstractmethod
    def save_model(self, directory: str):
        pass

    @abc.abstractmethod
    def delete_model(self):
        pass

    @abc.abstractmethod
    def load_model(self, directory: str):
        pass

    @abc.abstractmethod
    def fit(
            self,
            df_train: pd.DataFrame,
            df_validate: pd.DataFrame,
            df_test: pd.DataFrame,
    ):
        pass

    @abc.abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
