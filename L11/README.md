## Benchmark

Start the Triton server:
```shell
cd ../L10/triton
docker run -v `pwd`:/data --shm-size=1g --ulimit memlock=-1 -p 8000:8000 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:23.11-py3 make -C /data
```

Run the benchmark(single threaded serial requests):
```shell
python benchmark.py
```
The results should be similar to:
```
Latency, P25: 17.35ms
Latency, P50: 19.75ms
Latency, P75: 23.92ms
Latency, P99: 39.94ms
Avg RPS: 47.2
```
