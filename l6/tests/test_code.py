import tensorflow as tf
from l6 import trainer


def test_create_model():
  model = trainer.create_model()

  assert len(model.layers) > 2
  assert model.layers[0].compute_dtype == 'string'
  assert model.layers[-1].compute_dtype == 'float32'


def test_extract_loss():
  metrics = {'loss': tf.constant(10101)}

  result = trainer.extract_loss(metrics)

  assert result == {'loss': 10101}
