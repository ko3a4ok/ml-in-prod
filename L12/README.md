# Async inference

## Setup Kafka with Kserve
```shell
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.11/hack/quick_install.sh" | bash
kubectl create namespace kserve-test
kubectl config set-context --current --namespace=kserve-test
kubectl apply -f k8s/deployment.yaml
kubectl port-forward "$(kubectl get pod -o name | grep roma-model)" 8080:8080
```

Build the transformer container
```shell
docker build -t ko3a4ok/text-transformer:latest -f transformer.Dockerfile .
docker push ko3a4ok/text-transformer:latest
```
### Kafka
```shell
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install zookeeper bitnami/zookeeper --set replicaCount=1 --set auth.enabled=false --set allowAnonymousLogin=true --set persistance.enabled=false --version 11.0.0
helm install kafka bitnami/kafka --set zookeeper.enabled=false --set replicaCount=1 --set persistance.enabled=false --set logPersistance.enabled=false --set externalZookeeper.servers=zookeeper-headless.default.svc.cluster.local --version 21.0.0

kubectl apply -f https://github.com/knative/eventing/releases/download/v0.25.0/eventing-crds.yaml
kubectl apply -f https://github.com/knative/eventing/releases/download/v0.25.0/eventing-core.yaml
kubectl apply -f https://github.com/knative-sandbox/eventing-kafka/releases/download/v0.25.3/source.yaml

```

### Minio
```shell
kubectl apply -f k8s/minio.yaml
kubectl port-forward $(kubectl get pod --selector="app=minio" --output jsonpath='{.items[0].metadata.name}') 9000:9000
mc config host add myminio http://127.0.0.1:9000 minio minio123
mc mb myminio/input
mc mb myminio/output

mc admin config set myminio notify_kafka:1 tls_skip_verify="off"  queue_dir="" queue_limit="0" sasl="off" sasl_password="" sasl_username="" tls_client_auth="0" tls="off" client_tls_cert="" client_tls_key="" brokers="kafka-headless.default.svc.cluster.local:9092" topic="test" version=""
mc admin service restart myminio
mc event add myminio/input arn:minio:sqs::1:kafka -p --event put --suffix .txt
```

### Test
```shell
echo -e 'Hello\nGood bye' > sample.txt 
mc cp sample.txt myminio/input
```
