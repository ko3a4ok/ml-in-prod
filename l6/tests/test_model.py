import shutil
from pathlib import Path

import numpy

from l6 import trainer


def test_create_model():
  assert trainer.create_model()


def test_input_type():
  model = trainer.create_model()

  assert model.layers[0].compute_dtype == 'string'


def test_output_type():
  model = trainer.create_model()
  assert model.layers[-1].compute_dtype == 'float32'


def test_train_to_completion():
  path = Path('/tmp/mymodel')
  if path.exists():
    shutil.rmtree(path)
  path.mkdir()

  trainer.pipeline(path.absolute())

  assert (path / "checkpoint").exists()
  assert (path / "weights.data-00000-of-00001").exists()
  assert (path / "weights.index").exists()


def test_overfit():
  model = trainer.create_model()
  X = numpy.array(['ololo']*1000)
  y = numpy.array([1]*1000)
  data = {
      'excerpt': X,
      'target': y,
  }

  trainer.train(model, data)
  y_pred = model.predict(X)
  metrics = model.compute_metrics(x=None, y=y, y_pred=y_pred,
                                  sample_weight=None)
  assert metrics['loss'] < 0.05



