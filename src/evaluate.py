import os
from pathlib import Path
import yaml

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.model.project_model import ProjectModel


def plot_confusion_matrix(
        df: pd.DataFrame,
        path: str
):
    cm = confusion_matrix(df['Survived'], df['Survived_predictions'], normalize='true')
    ax = sns.heatmap(cm,
                     annot=True,
                     cmap="Blues",
                     fmt='.0%')

    filename = os.path.join(path, 'cm_of_survived.png')
    ax.figure.savefig(filename)
    ax.figure.clear()


def evaluate(df: pd.DataFrame, model: ProjectModel):
    df = model.predict(df)
    plot_confusion_matrix(df, './data/evaluate.dir')


if __name__ == '__main__':
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    data = pd.DataFrame(
        pd.read_hdf('data/train.dir/data.h5', index_col=0)
    )

    project_model = ProjectModel.load('data/train.dir/model.dir')

    Path('data/evaluate.dir').mkdir(parents=True, exist_ok=True)

    evaluate(df=data, model=project_model)
