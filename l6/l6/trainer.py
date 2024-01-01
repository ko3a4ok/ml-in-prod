import os

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, TextVectorization, \
  GlobalAveragePooling1D
import wandb


def create_vector():
  text_vectorization = TextVectorization(max_tokens=10000,
                                         output_mode='int',
                                         output_sequence_length=500)
  data = get_train_dataset()
  X = data['excerpt']
  text_vectorization.adapt([X])
  text_vectorization.save_assets('')


def create_model():
  text_vectorization = TextVectorization(max_tokens=10000,
                                         output_mode='int',
                                         output_sequence_length=500)
  text_vectorization.load_assets(os.path.dirname(os.path.abspath(__file__)))
  model = Sequential([
      text_vectorization,
      Embedding(10000, 128),
      GlobalAveragePooling1D(),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(1, activation='sigmoid')
  ])
  return model


def get_train_dataset():
  path = os.path.dirname(os.path.abspath(__file__))
  return pd.read_csv(os.path.join(path, 'data/train.csv'))


def get_test_dataset():
  path = os.path.dirname(os.path.abspath(__file__))
  return pd.read_csv(os.path.join(path, 'data/test.csv'))

def train(model, data):
  model.compile(optimizer='adam', loss='mse')

  X = data['excerpt']
  y = data['target']
  model.fit(X, y, epochs=3, batch_size=32, validation_split=0.2)


def extract_loss(metrics):
  loss = metrics['loss'].numpy()
  print('loss: ', loss)
  return {'loss' : loss}

def measure_quality(model):
  data = get_train_dataset()
  X = data['excerpt']
  y = data['target']
  y_pred = model.predict(X)
  metrics = model.compute_metrics(x=None, y=y, y_pred=y_pred,
                                  sample_weight=None)
  return extract_loss(metrics)


def save_into_wandb(model, model_path):
  with wandb.init(project="L6") as run:
    run.log(measure_quality(model))

    best_model = wandb.Artifact(f"model_{run.id}", type="model")
    best_model.add_file(os.path.join(model_path, 'weights.data-00000-of-00001'))
    best_model.add_file(os.path.join(model_path, 'weights.index'))
    best_model.add_file(os.path.join(model_path, 'checkpoint'))
    run.log_artifact(best_model)

    run.link_artifact(best_model, "model-registry/L6 Registered Model")

    run.finish()

def save_model(model, parent_dir=''):
  model.save_weights(os.path.join(parent_dir, 'weights'), overwrite=True, save_format=None, options=None)

  save_into_wandb(model, parent_dir)


def pipeline(parant_dir=''):
  model = create_model()
  train(model, get_train_dataset())
  measure_quality(model)
  save_model(model, parant_dir)


if __name__ == '__main__':
  pipeline()
