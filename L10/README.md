# Serving

## Seldon
Install heml
```shell
wget https://get.helm.sh/helm-v3.13.1-linux-amd64.tar.gz
tar zxvf helm-v3.13.1-linux-amd64.tar.gz 
sudo mv linux-amd64/helm /usr/local/bin/helm
```
Install s2i:
```shell
wget https://github.com/openshift/source-to-image/releases/download/v1.3.9/source-to-image-v1.3.9-574a2640-linux-amd64.tar.gz
tar zxvf source-to-image-v1.3.9-574a2640-darwin-amd64.tar.gz 
sudo mv s2i /usr/local/bin/
```

Install Istio
```shell
curl -L https://istio.io/downloadIstio | sh -
cd istio-X.YY.Z
export PATH=$PWD/bin:$PATH
istioctl install --set profile=demo -y
kubectl label namespace default istio-injection=enabled
ubectl apply -f - << END
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: seldon-gateway
  namespace: istio-system
spec:
  selector:
    istio: ingressgateway # use istio default controller
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"
END
```
Install Seldon
```shell
kubectl create namespace seldon-system
helm install seldon-core seldon-core-operator --repo https://storage.googleapis.com/seldon-charts --set usageMetrics.enabled=true --set istio.enabled=true --namespace seldon-system
helm install seldon-core seldon-core-operator --version 1.15.1 --repo https://storage.googleapis.com/seldon-charts --set usageMetrics.enabled=true --set ambassador.enabled=true  --namespace seldon-system

```

Containerize the model 
```shell
export DOCKER_API_VERSION=1.41
s2i build serving seldonio/seldon-core-s2i-python3:0.18 emb:0.1 -e MODEL_NAME=emb -e API_TYPE=REST -e SERVICE_TYPE=MODEL -e PERSISTENCE=0
kubectl create namespace seldon
kubectl apply -f k8s/seldon-deployment.yaml
```

Test
```shell
curl -X POST "http://IP:7777/seldon/seldon/seldon-model/api/v1.0/predictions" -H "accept: application/json" -H "Content-Type: application/json" -d '{"data": ["hello"]}'

```


## KServe

Install:
```shell
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.11/hack/quick_install.sh" | bash
kubectl create namespace kserve-test
kubectl apply -n kserve-test -f k8s/kserve-deployment.yaml
```

Create docker:
```shell
docker build . --tag ko3a4ok/roma-model:latest -f Dockerfile.kserve
docker push ko3a4ok/roma-model:latest
```

Run:
```shell
kubectl apply -n kserve-test -f k8s/kserve-deployment.yaml
```

Port forwarding:
```shell
kubectl port-forward "$(kubectl get pod -o name | grep roma-model)" 8080:8080
```

Test:
```shell
curl 'http://localhost:8080/v1/models/roma-model:predict'    -d '{"instances": ["Hello"]}' -H 'Content-type: application/json'
```
