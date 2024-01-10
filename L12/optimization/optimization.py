import time
from datetime import datetime

from sentence_transformers import ParallelSentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from sentence_transformers import losses
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, LoggingHandler, util, evaluation, models, InputExample
import logging
import os
import gzip
import csv
import random
import numpy as np
import torch
from torch.nn import Embedding, Linear
from torch.quantization import quantize_dynamic

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#We use AllNLI as a source of sentences for the distillation
nli_dataset_path = 'datasets/AllNLI.tsv.gz'

#Further, we use sentences extracted from the English Wikipedia to train the distillation
wikipedia_dataset_path = 'datasets/wikipedia-en-sentences.txt.gz'

#We use the STS benchmark dataset to see how much performance we loose
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

def download_datasets():
  if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)
  if not os.path.exists(wikipedia_dataset_path):
    util.http_get('https://sbert.net/datasets/wikipedia-en-sentences.txt.gz', wikipedia_dataset_path)

  if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

def dim_reduction(model, new_dimension = 128):
  # We measure the performance of the original model
  # and later we will measure the performance with the reduces dimension size
  logger.info("Read STSbenchmark test dataset")
  eval_examples = []
  with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
      if row['split'] == 'test':
        score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
        eval_examples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

  # Evaluate the original model on the STS benchmark dataset
  stsb_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(eval_examples, name='sts-benchmark-test')

  logger.info("Original model performance:")
  stsb_evaluator(model)

  ######## Reduce the embedding dimensions ########

  #Read sentences from NLI dataset
  nli_sentences = set()
  with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
      nli_sentences.add(row['sentence1'])
      nli_sentences.add(row['sentence2'])

  nli_sentences = list(nli_sentences)
  random.shuffle(nli_sentences)

  #To determine the PCA matrix, we need some example sentence embeddings.
  #Here, we compute the embeddings for 20k random sentences from the AllNLI dataset
  pca_train_sentences = nli_sentences[0:20000]
  train_embeddings = model.encode(pca_train_sentences, convert_to_numpy=True)

  #Compute PCA on the train embeddings matrix
  pca = PCA(n_components=new_dimension)
  pca.fit(train_embeddings)
  pca_comp = np.asarray(pca.components_)

  # We add a dense layer to the model, so that it will produce directly embeddings with the new size
  dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=new_dimension, bias=False, activation_function=torch.nn.Identity())
  dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
  model.add_module('dense', dense)

  # Evaluate the model with the reduce embedding size
  logger.info("Model with {} dimensions:".format(new_dimension))
  stsb_evaluator(model)
  return model

def distillation(teacher_model):
  output_path = "output/model-distillation-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

  # 1) Create a smaller student model by using only some of the teacher layers
  teacher_model.save('student')
  student_model = SentenceTransformer('student')

  auto_model = student_model._first_module().auto_model
  layers_to_keep = [1, 4, 7, 10]          #Keep 4 layers from the teacher
  logger.info("Remove layers from student. Only keep these layers: {}".format(layers_to_keep))
  new_layers = torch.nn.ModuleList([layer_module for i, layer_module in enumerate(auto_model.encoder.layer) if i in layers_to_keep])
  auto_model.encoder.layer = new_layers
  auto_model.config.num_hidden_layers = len(layers_to_keep)

  inference_batch_size = 512
  train_batch_size = 512



  #We need sentences to train our distillation. Here, we use sentences from AllNLI and from WikiPedia
  train_sentences_nli = set()
  dev_sentences_nli = set()

  train_sentences_wikipedia = []
  dev_sentences_wikipedia = []

  # Read ALLNLI
  with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
      if row['split'] == 'dev':
        dev_sentences_nli.add(row['sentence1'])
        dev_sentences_nli.add(row['sentence2'])
      else:
        train_sentences_nli.add(row['sentence1'])
        train_sentences_nli.add(row['sentence2'])

  train_sentences_nli = list(train_sentences_nli)
  random.shuffle(train_sentences_nli)
  train_sentences_nli = train_sentences_nli[:1000]

  dev_sentences_nli = list(dev_sentences_nli)
  random.shuffle(dev_sentences_nli)
  dev_sentences_nli = dev_sentences_nli[0:500] #Limit dev sentences to 5k

  # Read Wikipedia sentences file
  with gzip.open(wikipedia_dataset_path, 'rt', encoding='utf8') as fIn:
    wikipeda_sentences = [line.strip() for line in fIn]

  dev_sentences_wikipedia = wikipeda_sentences[0:1000] #Use the first 1k sentences from the wikipedia file for development
  train_sentences_wikipedia = wikipeda_sentences[1000:]
  train_sentences_wikipedia = random.choices(train_sentences_wikipedia, k=10000)


  # We use the STS benchmark dataset to measure the performance of student model im comparison to the teacher model
  logger.info("Read STSbenchmark dev dataset")
  dev_samples = []
  with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
      if row['split'] == 'dev':
        score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
        dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
  dev_samples = random.choices(dev_samples, k=len(dev_samples)//10)
  dev_evaluator_sts = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


  logger.info("Teacher Performance:")
  dev_evaluator_sts(teacher_model)


  # We train the student_model such that it creates sentence embeddings similar to the embeddings from the teacher_model
  # For this, we need a large set of sentences. These sentences are embedded using the teacher model,
  # and the student tries to mimic these embeddings. It is the same approach as used in: https://arxiv.org/abs/2004.09813
  train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model, batch_size=inference_batch_size, use_embedding_cache=False)
  train_data.add_dataset([[sent] for sent in train_sentences_nli], max_sentence_length=256)
  train_data.add_dataset([[sent] for sent in train_sentences_wikipedia], max_sentence_length=256)

  train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
  train_loss = losses.MSELoss(model=student_model)

  # We create an evaluator, that measure the Mean Squared Error (MSE) between the teacher and the student embeddings
  dev_sentences = dev_sentences_nli + dev_sentences_wikipedia
  dev_evaluator_mse = evaluation.MSEEvaluator(dev_sentences, dev_sentences, teacher_model=teacher_model)

  # Train the student model to imitate the teacher
  student_model.fit(train_objectives=[(train_dataloader, train_loss)],
                    evaluator=evaluation.SequentialEvaluator([dev_evaluator_sts, dev_evaluator_mse]),
                    epochs=1,
                    warmup_steps=100,
                    # evaluation_steps=1,
                    output_path=output_path,
                    save_best_model=True,
                    optimizer_params={'lr': 1e-4, 'eps': 1e-6},
                    use_amp=True)
  return student_model
def quantization(model):
  q_model = quantize_dynamic(model, {Linear})

  # Convert the dataset to a DataLoader ready for training
  logger.info("Read STSbenchmark dataset")
  test_samples = []
  sentences = []

  with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
      score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
      inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

      sentences.append(row['sentence1'])
      sentences.append(row['sentence2'])

      if row['split'] == 'test':
        test_samples.append(inp_example)

  sentences = sentences[0:10000]

  logger.info("Evaluating speed of unquantized model")
  start_time = time.time()
  emb = model.encode(sentences, show_progress_bar=True)
  diff_normal = time.time() - start_time
  logger.info("Done after {:.2f} sec. {:.2f} sentences / sec".format(diff_normal, len(sentences) / diff_normal))

  logger.info("Evaluating speed of quantized model")
  start_time = time.time()
  emb = q_model.encode(sentences, show_progress_bar=True)
  diff_quantized = time.time() - start_time
  logger.info("Done after {:.2f} sec. {:.2f} sentences / sec".format(diff_quantized, len(sentences) / diff_quantized))
  logger.info("Speed-up: {:.2f}".format(diff_normal / diff_quantized))
  #########

  evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')

  logger.info("Evaluate regular model")
  model.evaluate(evaluator)

  print("\n\n")
  logger.info("Evaluate quantized model")
  q_model.evaluate(evaluator)
  return q_model

def run():
  download_datasets()
  # model = SentenceTransformer('ko3a4ok/roma-model', device='cpu')
  model_name = 'all-distilroberta-v1'

  model = SentenceTransformer(model_name, device='cpu')
  model = distillation(model)
  model = dim_reduction(model)
  model.save('final')
  if model.device.type == 'cpu':
    model = quantization(model)


if __name__ == '__main__':
  run()
