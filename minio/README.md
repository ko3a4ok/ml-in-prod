
## Minio installation

### Local
```shell
wget https://dl.min.io/server/minio/release/linux-amd64/archive/minio_20231115204325.0.0_amd64.deb -O minio.deb
sudo dpkg -i minio.deb
mkdir ~/minio
minio server ~/minio --console-address :9090
```
### Docker
```shell
mkdir -p ~/minio

docker run \
-p 9000:9000 \
-p 9090:9090 \
--name minio \
-v ~/minio/data:/data \
-e "MINIO_ROOT_USER=minioadmin" \
-e "MINIO_ROOT_PASSWORD=minioadmin" \
quay.io/minio/minio server /data --console-address ":9090"

```
### K8S
```shell
curl https://raw.githubusercontent.com/minio/docs/master/source/extra/examples/minio-dev.yaml -O
```
NOTE: Update the `kubealpha.local` with the right value for your cluster. 

Use the follwing command to verify the corrent label for `kubernetes.io/hostname` 
```shell
 kubectl get nodes --show-labels
```

Enable port forwarding
```shell
kubectl port-forward pod/minio 9000 9090 -n minio-dev
```
