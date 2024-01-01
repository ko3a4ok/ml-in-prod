# Airflow pipelines

## Setup

Create cluster:
```shell
kind create cluster --name l8
kubectl create -f airflow-volumes.yaml
```
Install:
```shell
pip install apache-airflow
```

Startup Airflow webserver:
```shell
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow standalone
```

Set W&B api key:
```shell
airflow variables set wandb_api_key <WANDB_API_KEY>
```

Run the training DAG:
```shell
mkdir -p ~/airflow/dags/
cp training_dag.py ~/airflow/dags/
airflow dags unpause training
airflow dags trigger training
```

Run the inference DAG:
```shell
mkdir -p ~/airflow/dags/
cp inference_dag.py ~/airflow/dags/
airflow dags unpause inference
airflow dags trigger inference
```

