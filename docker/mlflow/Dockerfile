FROM ghcr.io/mlflow/mlflow:v2.19.0

RUN pip install --no-cache-dir psycopg2-binary

RUN mkdir -p /mlflow/artifacts && chmod 777 /mlflow/artifacts

EXPOSE 5000