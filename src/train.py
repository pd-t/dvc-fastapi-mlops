import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.model.project_model import ProjectModel


def train(
        df: pd. DataFrame,
        **kwargs
):
    temp, df_test = \
        train_test_split(df,
                         test_size=kwargs['data']['test_train_ratio'],
                         random_state=kwargs['data']['random_state']
                         )
    df_train, df_validate = \
        train_test_split(temp,
                         test_size=0.1,
                         random_state=kwargs['data']['random_state']
                         )

    project_model = ProjectModel(**params['model'])
    project_model.fit(df_train, df_validate, df_test)
    return df_train, df_test, project_model


if __name__ == '__main__':
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    data = pd.DataFrame(
        pd.read_hdf('data/prepare.dir/data.h5',
                    index_col=0)
    )

    Path('./data/train.dir').mkdir(parents=True, exist_ok=True)
    Path('./data/train.dir/model.dir').mkdir(parents=True, exist_ok=True)

    data_train, data_test, model = train(df=data, **params)

    model.save('./data/train.dir/model.dir')
    data_test.to_hdf('./data/train.dir/data.h5', key='df', mode='w')
