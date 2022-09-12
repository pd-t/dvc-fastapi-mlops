from pathlib import Path
import yaml
import pandas as pd
import urllib.request


def load(url: str) -> pd.DataFrame:
    print('load: download data')
    filename = "./data/load.dir/titanic.csv"
    urllib.request.urlretrieve(url, filename)
    print('load: read csv')
    df = pd.read_csv(filename)
    return df


if __name__ == '__main__':
    Path('data/load.dir').mkdir(parents=True, exist_ok=True)

    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    data = load(url=params['data']['url'])

    data.to_hdf('data/load.dir/data.h5', key='df', mode='w')
