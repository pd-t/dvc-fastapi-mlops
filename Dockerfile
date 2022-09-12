FROM python:3.8.10-slim AS base
RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/* \
&& pip install poetry==1.1.15

FROM base AS python-environment
COPY *.toml *.lock ./
RUN poetry config virtualenvs.create false && poetry install
WORKDIR /app

FROM python-environment AS dvc-repro
COPY . .
RUN dvc init -f && dvc repro

FROM python-environment AS app
COPY . .
COPY --from=dvc-repro /app/data/train.dir/model.dir /app/data/train.dir/model.dir
EXPOSE 8080
ENTRYPOINT uvicorn --host 0.0.0.0 --workers 1 --port 8080 "app.main:app"
