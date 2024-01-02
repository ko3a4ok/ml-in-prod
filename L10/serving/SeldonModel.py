import logging

from predictor import Predictor

logger = logging.getLogger()

class SeldonModel:
  def __init__(self):
    self.predictor = Predictor()

  def predict(self, X):
    logger.info("Input: ", X)
    return self.predictor.predict(X)
