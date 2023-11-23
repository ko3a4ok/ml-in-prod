import tensorflow as tf
import pandas as pd
import wandb
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, TextVectorization, \
  GlobalAveragePooling1D

data = pd.read_csv('train.csv')
X = data['excerpt']
y = data['target']
vectorize_layer = TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=5000)
vectorize_layer.adapt([X])
model = Sequential([
    vectorize_layer,
    Embedding(50000, 128),
    GlobalAveragePooling1D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

CONFIG = {
    "epochs": 32,
    "validation_split": 0.2,
    "batch_size": 16,
}
EXTRA_CONFIG = {
    "learning_rate": 0.0001,
}
wandb.init(
    project="L5",
    config=CONFIG,
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=EXTRA_CONFIG['learning_rate']), loss='mse')


class WandbCallback(Callback):
  def on_epoch_end(self, epoch, logs=None):
    wandb.log(logs)


model.fit(X, y, **CONFIG, callbacks=[WandbCallback()])
wandb.finish()
