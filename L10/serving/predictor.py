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
    encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    model_output = self.model(**encoded_input)
    sentence_embeddings = model_output[0][:, 0]
    return torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).numpy()


if __name__ == '__main__':
    predictor = Predictor()
    print(predictor.predict(['hello']))
