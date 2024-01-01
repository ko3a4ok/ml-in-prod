from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()


class Prediction(BaseModel):
  instances: list


class Response(BaseModel):
  embeddings: list


model = SentenceTransformer('TaylorAI/bge-micro')


@app.post('/predict')
def predict(prediction: Prediction):
  return Response(embeddings=model.encode(prediction.instances).tolist())
