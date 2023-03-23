import pytest
from aniemore.datasets import Dataset


def test_dataset_instantiation():
    _name = 'Russian Emotional Speech Dialoges'
    _url = 'aniemore/resd'
    test_dataset = Dataset(_name, _url)
    assert test_dataset.dataset_name == _name
    assert test_dataset.dataset_url == _url
