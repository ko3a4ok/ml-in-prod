import io
import os

from minio import Minio


class CrudClient(object):
  def __init__(self, bucket):
    self._client = Minio('localhost:9000', secure=False,
                         access_key=os.getenv('MINIO_LOGIN', 'minioadmin'),
                         secret_key=os.getenv('MINIO_PASSWORD', 'minioadmin'))
    self._bucket = bucket
    if not self._client.bucket_exists(self._bucket):
      self._client.make_bucket(self._bucket)

  def upload(self, filename: str, data: bytes):
    self._client.put_object(self._bucket, filename, io.BytesIO(data), len(data))

  def remove(self, filename: str):
    self._client.remove_object(self._bucket, filename)

  def download(self, filename):
    obj = self._client.get_object(self._bucket, filename)
    return obj.data

  def get_files(self):
    return [obj.object_name for obj in
            self._client.list_objects(self._bucket, recursive=True)]


if __name__ == '__main__':
  client = CrudClient('zig')
  print(client.remove(client.get_files()[0]))
  print(client.get_files())
  print(client.download(client.get_files()[0]))
