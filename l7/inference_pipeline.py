import os

from kfp import dsl
from kfp.dsl import Dataset
from kfp.dsl import Input
from kfp.dsl import Output

IMAGE = "ko3a4ok/zbs:latest"


@dsl.component(packages_to_install=["gdown"])
def load_data(test_data: Output[Dataset]):
  import gdown
  import zipfile
  from pathlib import Path
  import shutil
  gdown.download(
      'https://drive.google.com/file/d/1QFrdPSbLyhtkAgnecj0C98veiEbC3CeM/view',
      fuzzy=True)
  with zipfile.ZipFile('commonlitreadabilityprize.zip', 'r') as zip_ref:
    zip_ref.extractall('/data')

  shutil.move(Path("/data") / "test.csv", test_data.path)


@dsl.component(base_image=IMAGE)
def download_model(weight_data: Output[Dataset],
    weight_index: Output[Dataset], vocabulary: Output[Dataset]):
  import shutil
  from zbs.inference import download_model

  download_model()

  shutil.move('model/weights.data-00000-of-00001', weight_data.path)
  shutil.move('model/weights.index', weight_index.path)
  shutil.move('model/vocabulary.txt', vocabulary.path)


@dsl.component(base_image=IMAGE, packages_to_install=['pandas'])
def inference(test_data: Input[Dataset], weight_data: Input[Dataset],
    weight_index: Input[Dataset], vocabulary: Input[Dataset],
    result_file: Output[Dataset]):
  import shutil
  import pandas as pd
  from zbs.inference import inference

  shutil.move(weight_data.path, 'weights.data-00000-of-00001')
  shutil.move(weight_index.path, 'weights.index')
  shutil.move(vocabulary.path, 'vocabulary.txt')

  result = inference(pd.read_csv(test_data.path))

  result.to_csv(result_file.path)


@dsl.component(base_image=IMAGE)
def upload(result_file: Input[Dataset]) -> str:
  from zbs.inference import save_into_wandb

  save_into_wandb(result_file.path)
  return 'Done'


@dsl.pipeline
def inference_pipeline() -> str:
  load_data_task = load_data()
  download_model_task = download_model()
  download_model_task.set_env_variable(name="WANDB_API_KEY",
                                       value=os.getenv("WANDB_API_KEY"))
  inference_task = inference(**load_data_task.outputs,
                             **download_model_task.outputs)
  upload_task = upload(**inference_task.outputs)
  upload_task.set_env_variable(name="WANDB_API_KEY",
                               value=os.getenv("WANDB_API_KEY"))
  return upload_task.output


if __name__ == '__main__':
  from kfp import compiler

  compiler.Compiler().compile(inference_pipeline, 'pipeline.yaml')
  from kfp.client import Client

  client = Client(host=os.getenv('KUBERFLOW_URI', 'http://localhost:8081'))
  run = client.create_run_from_pipeline_package(
      'pipeline.yaml',
      arguments={},
  )
