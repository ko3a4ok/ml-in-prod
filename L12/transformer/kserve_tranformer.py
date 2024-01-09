import logging
from typing import Dict, Union

import boto3
import argparse

import kserve
from kserve import InferRequest, InferResponse
from kserve.protocol.grpc.grpc_predict_v2_pb2 import ModelInferResponse

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

session = boto3.Session()
client = session.client('s3', endpoint_url='http://minio-service:9000', aws_access_key_id='minio',
                        aws_secret_access_key='minio123')


def text_transform(text):
  return [i.strip()[:512] for i in text]



class TextEmbedderTransformer(kserve.Model):
  def __init__(self, name: str, predictor_host: str):
    super().__init__(name)
    self.predictor_host = predictor_host
    self._key = None

  async def preprocess(self, inputs: Union[Dict, InferRequest],
      headers: Dict[str, str] = None) -> Union[Dict, InferRequest]:
    logging.info("Received inputs %s", inputs)
    if inputs['EventName'] == 's3:ObjectCreated:Put':
      bucket = inputs['Records'][0]['s3']['bucket']['name']
      key = inputs['Records'][0]['s3']['object']['key']
      self._key = key
      request = text_transform(client.get_object(Bucket=bucket,Key=key)['Body'].read().decode('utf-8'))
      return {"instances": request}
    raise Exception("unknown event")

  async def postprocess(
      self,
      response: Union[Dict, InferResponse, ModelInferResponse],
      headers: Dict[str, str] = None
  ) -> Union[Dict, ModelInferResponse]:
    logging.info("response: %s", response)
    upload_path = f'emb/{self._key}'
    data = '\t'.join(map(str, response["predictions"]))
    logging.info('Uploading: ', data)
    client.put_object(Bucket="output", Key=upload_path, Body=data)
    logging.info(f"Text {self._key} successfully uploaded to {upload_path}")
    return response


parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument('--model_name', default="roma-model",
                    help='The name that the model is served under.')
parser.add_argument('--predictor_host', help='The URL for the model predict function', required=True)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
  transformer = TextEmbedderTransformer(args.model_name, predictor_host=args.predictor_host)
  server = kserve.ModelServer()
  server.start(models=[transformer])
