import tensorflow as tf
import pandas as pd
from wandb.integration.keras import WandbCallback

import wandb
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, TextVectorization, \
  GlobalAveragePooling1D


def create_model(X):
  vectorize_layer = TextVectorization(
      max_tokens=10000,
      output_mode='int',
      output_sequence_length=5000)
  vectorize_layer.adapt([X])
  return Sequential([
      vectorize_layer,
      Embedding(50000, 128),
      GlobalAveragePooling1D(),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(1, activation='sigmoid')
  ])


def get_data():
  data = pd.read_csv('train.csv')
  return data['excerpt'], data['target']


def train():
  wandb.init(
      project="L5",
  )
  X, y = get_data()
  config = wandb.config
  print("Mega config: ", config)

  optimizer = None
  if config.optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
  elif config.optimizer == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)

  model = create_model(X)
  model.compile(optimizer=optimizer, loss='mse')
  model.fit(X, y, epochs=config.epochs,
            validation_split=config.validation_split,
            batch_size=config.batch_size, callbacks=[WandbCallback()])
  wandb.finish()


if __name__ == '__main__':
  train()
