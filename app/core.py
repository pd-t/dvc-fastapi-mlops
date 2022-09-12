import pandas as pd
from .models import RequestExample, ResponseExample
from src.prepare import prepare
from src.model.project_model import ProjectModel

project_model = ProjectModel.load('./data/train.dir/'
                                  'model.dir')


def load(body: RequestExample) -> pd.DataFrame:
    data = {
        'Age': body.age,
        'Sex': body.sex,
        'Pclass': body.pclass
    }
    df = pd.DataFrame.from_dict([data])
    return df


def predict(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare(df)
    df = project_model.predict(df)
    return df


def post_example_core(
        body: RequestExample
) -> ResponseExample:
    df = load(body)
    df = prepare(df)
    df = predict(df)
    response = ResponseExample.construct(
        survived=str(df['Survived_predictions'][0]),
        result_type='omt_answers',
    )
    return response
