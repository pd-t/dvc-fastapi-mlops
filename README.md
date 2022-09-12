# dvc-fastapi-mlops

One of the main reasons for poor performance of AI models in the
Use is the deviation of the data pipeline in the training and the 
later application. So what is the mlops problem about?

When creating an AI model, a data scientist or ml engineering prepares the 
data in order to achieve the best possible result. It is common practice 
that the model is directly released for inference. The worst case, when 
no information about the data preprocessing is passed on to a developer 
which uses the model. This is one reason why models degenerate and 
disappoint in practice.

Combining DvC and FastAPI is an option how this problem can be solved. The 
basic idea is that the data go through exactly the same path of 
preprocessing and through the model during training and later application. 
This is achieved by loading the incoming inference data into a 
dataframes, which reflects the dataframe of the loaded training data before 
preprocessing. Strictly speaking, the creation of the RestAPI could be 
automated, based on the data frame structure at the beginning and end of 
the data pipeline.

In order to make the example not too complex API security features, GPU 
support and CI/CD/CML code are not included.

## Getting started
The complete example is dockerized, so if you want to test the server just 
write

```sh
$ docker-compose up app
```
and open your browser at `http://0.0.0.0:8080/docs`.

## Development

Do you like more? Well, a poetry environment is provided. To activate the 
environment, use
```sh
$ poetry install
```
and
```sh
$ poetry shell
```

The FastAPI code can be generated by
```sh
$ fastapi-codegen --input app/openapi/openapi.yaml -t app/openapi/jinja --output app
```
