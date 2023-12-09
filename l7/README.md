# Deploying Kubeflow Pipelines

1. Deploy the Kubeflow Pipelines:
```shell
export PIPELINE_VERSION=2.0.3
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
```

2. Port forwarding for Kuberflow
```shell
kubectl port-forward --address=0.0.0.0 svc/ml-pipeline-ui 8888:80 -n kubeflow

```
3. Set env variables:
```shell
export WANDB_API_KEY=****************
export KUBERFLOW_URI=http://localhost:8081
```
4. Run training pipeline
```shell
python training_pipeline.py
```
5. Run inference pipeline
```shell
python inference_pipeline.py
```
