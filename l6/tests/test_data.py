import pytest

from l6 import trainer
from great_expectations.dataset.pandas_dataset import PandasDataset


@pytest.fixture(scope="session")
def train_data() -> PandasDataset:
  return PandasDataset(trainer.get_train_dataset())


@pytest.fixture(scope="session")
def test_data() -> PandasDataset:
  return PandasDataset(trainer.get_test_dataset())


def test_data_shape(train_data, test_data):
  assert train_data.shape == (2834, 6)
  assert test_data.shape == (7, 4)


def test_data_columns(train_data):
  assert train_data.expect_column_to_exist('excerpt')['success']
  assert train_data.expect_column_to_exist('target')['success']


def test_input_content(train_data, test_data):
  assert train_data.expect_column_values_to_not_be_null('excerpt')['success']
  assert train_data.expect_column_values_to_be_unique('excerpt')['success']

  assert test_data.expect_column_values_to_not_be_null('excerpt')['success']
  assert test_data.expect_column_values_to_be_unique('excerpt')['success']


def test_target(train_data):
  assert train_data.expect_column_values_to_be_between('target', min_value=-5,
                                                       max_value=5)['success']
