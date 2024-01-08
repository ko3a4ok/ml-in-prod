import random
import time

import pandas as pd
import requests
DATA_URL = 'https://gist.github.com/ko3a4ok/9e8128ca2917a2b9379a716b50ef621c/raw/0df0af04199f0cfcfe6fb7531f467bbbaf0e183e/cymbal_product_desc.txt'
MODEL_URL = 'http://localhost:8000/v2/models/ololo/versions/1/infer'
PROB_NUM = 1000


def print_report(latencies):
  latencies.sort()
  n = len(latencies)
  for p in [25, 50, 75, 99]:
    print(f'Latency, P{p}: {1000*latencies[int(p*n/100)]:.2f}ms')
  avg_latency = sum(latencies)/n
  avg_rps = 1/avg_latency
  print(f'Avg RPS: {avg_rps:.1f}')


def run():
  data = pd.read_fwf(DATA_URL).iloc[:,0].tolist()
  latencies = []
  for i in range(PROB_NUM):
    txt = random.choice(data)[:512]
    input = {"inputs": [{"name": "text", "datatype": "BYTES", "shape": [1, 1], "data": [txt]}]}
    start = time.time()
    resp = requests.post(MODEL_URL, json=input)
    assert resp.status_code == 200
    if resp.status_code != 200:
      print(txt)
    latencies.append(time.time() - start)

  print_report(latencies)


if __name__ == '__main__':
  run()
