import unittest

from client import CrudClient


class MyTestCase(unittest.TestCase):
  def setUp(self) -> None:
    self.client = CrudClient('test')

  def tearDown(self) -> None:
    for file in self.client.get_files():
      self.client.remove(file)

  def test_create_object(self):
    self.client.upload('file.txt', b'Simple content')
    self.assertListEqual(self.client.get_files(), ['file.txt'])

  def test_download(self):
    self.client.upload('f.txt', b'Very simple content')
    self.assertEqual(self.client.download('f.txt'), b'Very simple content')

  def test_delete(self):
    self.client.upload('f.txt', b'File to remove')
    self.assertListEqual(self.client.get_files(), ['f.txt'])
    self.client.remove('f.txt')
    self.assertFalse(self.client.get_files())


if __name__ == '__main__':
  unittest.main()
