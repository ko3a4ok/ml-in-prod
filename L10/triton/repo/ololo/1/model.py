import triton_python_backend_utils as pb_utils

from typing import List

import torch
from transformers import AutoModel
from transformers import AutoTokenizer


class Predictor:
  def __init__(self, model_load_path='TaylorAI/bge-micro'):
    self.tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    self.model = AutoModel.from_pretrained(model_load_path)
    self.model.eval()

  @torch.no_grad()
  def predict(self, text: List[str]):
    encoded_input = self.tokenizer(text, padding=True, truncation=True,
                                   return_tensors='pt')
    model_output = self.model(**encoded_input)
    sentence_embeddings = model_output[0][:, 0]
    return torch.nn.functional.normalize(sentence_embeddings, p=2,
                                         dim=1).numpy()


class TritonPythonModel:
  def initialize(self, args):
    self.predictor = Predictor()

  def execute(self, requests):
    responses = []
    for request in requests:
      text = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()[0][0]
      text = str(text, 'utf-8')
      out_output = pb_utils.Tensor("embedding", self.predictor.predict(text))
      inference_response = pb_utils.InferenceResponse(output_tensors=[
          out_output
      ])
      responses.append(inference_response)
    return responses
