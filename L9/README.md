# Model Serving

Build docker
```shell
docker build --tag serving:latest .
docker tag serving:latest ko3a4ok/serving:latest
docker push ko3a4ok/serving:latest
```

Deploy:
```shell
kubectl create -f deployment.yaml
```

Port forward:
```shell
kubectl port-forward svc/web 8501:8501
kubectl port-forward svc/api 8000:8000
```
