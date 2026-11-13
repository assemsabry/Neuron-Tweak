import pytest

from neuron_tweak.analysis import AnalysisError
from neuron_tweak.safety import parse_layer_selection


def test_parse_layer_selection_all():
    assert parse_layer_selection(None, 32) is None
    assert parse_layer_selection("", 32) is None
    assert parse_layer_selection("all", 32) is None
    assert parse_layer_selection("ALL", 32) is None


def test_parse_layer_selection_list_and_ranges():
    assert parse_layer_selection("1,3,5-7", 10) == [1, 3, 5, 6, 7]


def test_parse_layer_selection_deduplicates():
    assert parse_layer_selection("2,2,1-2", 5) == [1, 2]


def test_parse_layer_selection_rejects_out_of_range():
    with pytest.raises(AnalysisError):
        parse_layer_selection("0", 10)
    with pytest.raises(AnalysisError):
        parse_layer_selection("11", 10)
    with pytest.raises(AnalysisError):
        parse_layer_selection("5-3", 10)


def test_parse_layer_selection_rejects_garbage():
    with pytest.raises(AnalysisError):
        parse_layer_selection("abc", 10)
    with pytest.raises(AnalysisError):
        parse_layer_selection("1-abc", 10)
