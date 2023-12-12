from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()


class Prediction(BaseModel):
  instances: list[str]


class Response(BaseModel):
  embeddings: list[list[float]]


model = SentenceTransformer('TaylorAI/bge-micro')


@app.post('/model:predict')
def predict(prediction: Prediction):
  return Response(embeddings=model.encode(prediction.instances).tolist())
