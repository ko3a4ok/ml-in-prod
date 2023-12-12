from fastapi.testclient import TestClient
import numpy as np

from context import api

client = TestClient(api.app)

HELLO_EMBEDDING = api.model.encode('hello').tolist()
BYE_EMBEDDING = api.model.encode('bye-bye').tolist()


def test_empty_input():
  response = client.post("/predict", json={})
  assert response.status_code == 422


def test_empty_predictions():
  response = client.post("/predict", json={'instances': []})
  assert response.status_code == 200
  assert response.json() == {'embeddings': []}


def test_single_predictions():
  response = client.post("/predict", json={'instances': ['hello']})
  assert response.status_code == 200
  assert response.json() == {'embeddings': [HELLO_EMBEDDING]}


def test_batch_predictions():
  response = client.post("/predict",
                         json={'instances': ['hello', 'bye-bye']})
  assert response.status_code == 200
  data = response.json()
  assert len(data['embeddings']) == 2

  assert all(np.isclose(HELLO_EMBEDDING, data['embeddings'][0], rtol=1.e-3))
  assert all(np.isclose(BYE_EMBEDDING, data['embeddings'][1], rtol=1.e-3))
