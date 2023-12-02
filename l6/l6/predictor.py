from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, TextVectorization, \
  GlobalAveragePooling1D


class Predictor:
  def __init__(self, path):
    vectorize_layer = TextVectorization(
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=5000)
    vectorize_layer.load_assets('')
    self.model = Sequential([
        vectorize_layer,
        Embedding(50000, 128),
        GlobalAveragePooling1D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    self.model.load_weights(path)

  def predict(self, X):
    return self.model.predict(X)
