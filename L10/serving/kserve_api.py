from predictor import Predictor
from typing import Dict
from kserve import Model, ModelServer

class CustomModel(Model):
  def __init__(self, name: str):
    super().__init__(name)
    self.name = name
    self.load()

  def load(self):
    self.predictor = Predictor()
    self.ready = True

  def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
    instances = payload["instances"]
    predictions = self.predictor.predict(instances).tolist()
    return {"predictions": predictions}

if __name__ == "__main__":
  model = CustomModel("roma-model")
  ModelServer(enable_docs_url=True).start([model])
