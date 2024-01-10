# Model inference optimization

## Model distillation

This technique allowed to improve the latency by 2x, with the quality loss of ~
3%

## Dimensionality reduction

This approach does help to improve the latency at all. The base model is 768,
and the output embedding model is 128.
This technique is good for the storage reduction, but it doesn't help for the
real time inference.

## Quantization

If supported only for CPU-based workload. This technique allowed to improve the
latency by 30% with the acceptable loss of quality. 
