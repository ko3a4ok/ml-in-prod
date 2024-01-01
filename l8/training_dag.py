from airflow.models import Variable
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

default_args = {
    'depends_on_past': False,
}

volume = k8s.V1Volume(
    name="training-storage",
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name="training-storage"),
)
volume_mount = k8s.V1VolumeMount(name="training-storage", mount_path="/tmp/", sub_path=None)

@dag(
    dag_id="training",
    description="Training ML model DAG",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["training"],
)
def training():
  download_data_task = KubernetesPodOperator(
      task_id="download-data",
      image="ko3a4ok/zbs:latest",
      cmds=["python", "zbs/cli.py", "download-data", '--path=/tmp/data'],
      in_cluster=False,
      is_delete_operator_pod=False,
      namespace="default",
      volumes=[volume],
      volume_mounts=[volume_mount],
  )
  training_task = KubernetesPodOperator(
      task_id="training",
      image="ko3a4ok/zbs:latest",
      cmds=["python", "zbs/cli.py", "train", "--data-path=/tmp/data", '--model-path=/tmp/model'],
      in_cluster=False,
      is_delete_operator_pod=False,
      namespace="default",
      volumes=[volume],
      volume_mounts=[volume_mount],
  )

  upload_task = KubernetesPodOperator(
      task_id="upload_model",
      image="ko3a4ok/zbs:latest",
      cmds=["python", "zbs/cli.py", "upload-model", "--path=/tmp/model"],
      env_vars={"WANDB_API_KEY": Variable.get("wandb_api_key")},
      in_cluster=False,
      is_delete_operator_pod=False,
      namespace="default",
      volumes=[volume],
      volume_mounts=[volume_mount],
  )

  download_data_task >> training_task >> upload_task

d = training()
