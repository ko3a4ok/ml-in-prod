from airflow.models import Variable
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.providers.cncf.kubernetes.operators.pod import \
  KubernetesPodOperator
from kubernetes.client import models as k8s

default_args = {
    'depends_on_past': False,
}

volume = k8s.V1Volume(
    name="inference-storage",
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name="inference-storage"),
)
volume_mount = k8s.V1VolumeMount(name="inference-storage", mount_path="/tmp/",
                                 sub_path=None)


@dag(
    dag_id="inference",
    description="Inference ML model DAG",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["inference"],
)
def inference():
  default_params = {
      'image': "ko3a4ok/zbs:latest",
      'in_cluster': False,
      'namespace': "default",
      'volumes': [volume],
      'volume_mounts': [volume_mount],
  }

  @task
  def task_1():
    x = 2
    print('Running task #1 ', Variable.get('wandb_api_key'))

  clean_storage_before_start = KubernetesPodOperator(
      task_id="clean_storage_before_start",
      cmds=["rm", "-rf", "/tmp/*"],
      **default_params,
  )
  download_data_task = KubernetesPodOperator(
      task_id="download-data",
      cmds=["python", "zbs/cli.py", "download-data", '--path=/tmp/data'],
      **default_params,
  )
  load_model_task = KubernetesPodOperator(
      task_id="download-model",
      cmds=["python", "zbs/cli.py", "download-model", "--path=/tmp"],
      env_vars={"WANDB_API_KEY": Variable.get("wandb_api_key")},
      **default_params,
  )

  inference_task = KubernetesPodOperator(
      task_id="inference-task",
      cmds=["python", "zbs/cli.py", "inference-data",
            "--data-path=/tmp/data", '--model-path=/tmp/model'],
      **default_params,
  )

  upload_results_task = KubernetesPodOperator(
      task_id="upload-results",
      cmds=["python", "zbs/cli.py", "upload-results", "--path=/tmp/data"],
      env_vars={"WANDB_API_KEY": Variable.get("wandb_api_key")},
      **default_params,
  )

  # Tasks
  task_1() >> clean_storage_before_start >> [download_data_task,
                                             load_model_task] >> inference_task >> upload_results_task


d = inference()
