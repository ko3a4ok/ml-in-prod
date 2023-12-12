from fastapi.testclient import TestClient

from context import api
client = TestClient(api.app)
HELLO_EMBEDDING = api.model.encode('hello again').tolist()
BYE_EMBEDDING = api.model.encode('bye-bye').tolist()


def test_empty_input():
  response = client.post("/model:predict", json={})
  assert response.status_code == 422


def test_empty_predictions():
  response = client.post("/model:predict", json={'instances': []})
  assert response.status_code == 200
  assert response.json() == {'embeddings': []}


def test_single_predictions():
  response = client.post("/model:predict", json={'instances': ['hello again']})
  assert response.status_code == 200
  assert response.json() == {'embeddings': [HELLO_EMBEDDING]}


def test_batch_predictions():
  response = client.post("/model:predict",
                         json={'instances': ['hello again', 'bye-bye']})
  assert response.status_code == 200
  assert response.json() == {'embeddings': [HELLO_EMBEDDING, BYE_EMBEDDING]}
