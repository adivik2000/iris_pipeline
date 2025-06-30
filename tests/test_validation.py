import pandas as pd
import pytest

@pytest.fixture
def iris_data():
    return pd.read_csv('data/iris.csv')

def test_shape(iris_data):
    assert iris_data.shape[1] == 5

def test_no_nulls(iris_data):
    assert iris_data.isnull().sum().sum() == 0

def test_species_labels(iris_data):
    expected = {"setosa", "versicolor", "virginica"}
    actual = set(iris_data["species"].unique())
    assert actual == expected