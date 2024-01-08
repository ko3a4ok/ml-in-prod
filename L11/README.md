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



## HPA
### Install metric server:
```shell
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch -n kube-system deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

```

### Start the cluster
```shell
kubectl apply -f k8s/deployment.yaml
kubectl port-forward --address 0.0.0.0 svc/emb-api 8000:8000
```


### Simulate the load from the cluster
```shell
while sleep 1 ; do curl -X POST emb-api.default.svc.cluster.local:8000/predict -H 'Content-Type: application/json' -d '{"instances": ["Hello"]}' &  done
```

### Track the result:
```shell
kubectl get hpa emb-api --watch
```
The result should be something like:
```shell
NAME      REFERENCE            TARGETS   MINPODS   MAXPODS   REPLICAS   AGE
emb-api   Deployment/emb-api   53%/50%   1         10        3          61m
emb-api   Deployment/emb-api   156%/50%   1         10        3          61m
emb-api   Deployment/emb-api   86%/50%    1         10        6          61m
emb-api   Deployment/emb-api   149%/50%   1         10        10         62m
emb-api   Deployment/emb-api   138%/50%   1         10        10         62m
emb-api   Deployment/emb-api   86%/50%    1         10        10         62m
emb-api   Deployment/emb-api   30%/50%    1         10        10         62m
```
