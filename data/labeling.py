import json
import random

import label_studio_sdk


def generate_labeling_data():
  with open('data_raw.json') as f:
    input_data = json.load(f)
  output_data = []
  answers = [q['query'] for q in input_data]
  for q in input_data:
    answer1 = q['query']
    while True:
      answer2 = random.choice(answers)
      if answer1 != answer2:
        break
    if random.random() < 0.5:
      answer1, answer2 = answer2, answer1
    output_data.append({
        'schema': q['schema'],
        'request': q['request'],
        'query1': answer1,
        'query2': answer2,
    })
  return output_data


ls = label_studio_sdk.Client('http://localhost:8080/',
                             'a0f445d34da93af75b9ff1ad3807157f0c5041d9')
with open('label_config.xml', 'r') as f:
  label_config = f.read()

project = ls.start_project(title="Very great labeling project",
                           label_config=label_config)
project.import_tasks(generate_labeling_data())
