import logging
from ludwig.api import LudwigModel
import pandas as pd

from src.model.template.ludwig import LudwigTemplate


class TitanicModel(LudwigTemplate):
    def __init__(
            self,
            epochs: int = None,
            early_stop: int = None,
            **kwargs
    ):
        super().__init__()
        self.epochs = epochs
        self.early_stop = early_stop

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate(self):
        model_definition = {
            'input_features': [],
            'output_features': [],
            'training': {
                'epochs': self.epochs,
                'early_stop': self.early_stop
            },
        }

        input_feature_block = {
            'name': 'Pclass',
            'type': 'category'
        }
        model_definition['input_features'].append(input_feature_block)

        input_feature_block = {
            'name': 'Sex',
            'type': 'category'
        }
        model_definition['input_features'].append(input_feature_block)

        input_feature_block = {
            'name': 'Age',
            'type': 'number',
            'preprocessing':
                {
                    'missing_value_strategy': 'fill_with_mean'
                }
        }
        model_definition['input_features'].append(input_feature_block)

        output_features = {
            'name': 'Survived',
            'type': 'binary',
        }
        model_definition['output_features'].append(output_features)

        return LudwigModel(model_definition, logging_level=logging.INFO)
