import os

from kfp import dsl
from kfp.dsl import Dataset
from kfp.dsl import Input
from kfp.dsl import Output

IMAGE = "ko3a4ok/zbs:latest"


@dsl.component(packages_to_install=["gdown"])
def load_data(train_data: Output[Dataset], test_data: Output[Dataset]):
  import gdown
  import zipfile
  from pathlib import Path
  import shutil
  gdown.download(
      'https://drive.google.com/file/d/1QFrdPSbLyhtkAgnecj0C98veiEbC3CeM/view',
      fuzzy=True)
  with zipfile.ZipFile('commonlitreadabilityprize.zip', 'r') as zip_ref:
    zip_ref.extractall('/data')

  shutil.move(Path("/data") / "train.csv", train_data.path)
  shutil.move(Path("/data") / "test.csv", test_data.path)


@dsl.component(base_image=IMAGE, packages_to_install=['pandas'])
def train(train_data: Input[Dataset], weight_data: Output[Dataset],
    weight_index: Output[Dataset], vocabulary: Output[Dataset]):
  import shutil
  import pandas as pd
  from zbs.trainer import train

  train(pd.read_csv(train_data.path))

  shutil.move('weights.data-00000-of-00001', weight_data.path)
  shutil.move('weights.index', weight_index.path)
  shutil.move('vocabulary.txt', vocabulary.path)


@dsl.component(base_image=IMAGE)
def upload(weight_data: Input[Dataset],
    weight_index: Input[Dataset], vocabulary: Input[Dataset]) -> str:
  from zbs.trainer import save_into_wandb
  import shutil

  shutil.move(weight_data.path, 'weights.data-00000-of-00001')
  shutil.move(weight_index.path, 'weights.index')
  shutil.move(vocabulary.path, 'vocabulary.txt')
  save_into_wandb('')
  return 'Done'


@dsl.pipeline
def train_pipeline() -> str:
  data = load_data()
  train_task = train(train_data=data.outputs['train_data'])
  upload_task = upload(**train_task.outputs)
  upload_task.set_env_variable(name="WANDB_API_KEY", value=os.getenv("WANDB_API_KEY"))
  return upload_task.output


if __name__ == '__main__':
  from kfp import compiler

  compiler.Compiler().compile(train_pipeline, 'pipeline.yaml')
  from kfp.client import Client


  client = Client(host=os.getenv('KUBERFLOW_URI', 'http://localhost:8081'))
  run = client.create_run_from_pipeline_package(
      'pipeline.yaml',
      arguments={},
  )
