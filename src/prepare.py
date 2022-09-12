from pathlib import Path
import pandas as pd
import yaml


def prepare(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # Do some fancy preprocessing here if necessary...
    return df


if __name__ == '__main__':
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    data = pd.DataFrame(
        pd.read_hdf('data/load.dir/data.h5', index_col=0)
    )

    Path('data/prepare.dir').mkdir(parents=True, exist_ok=True)

    data = prepare(
        df=data,
        **params['data']
    )

    data.to_hdf('data/prepare.dir/data.h5', key='df', mode='w')
