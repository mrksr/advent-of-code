import numpy as np
import pytest

from aoc.day_08 import inputs, main


def test_parse():
    input = inputs.multi_step()[1]
    moves, states = main.parse(input)
    assert np.array_equal(moves, np.array([0, 0, 1]))
    assert states == {
        "AAA": ("BBB", "BBB"),
        "BBB": ("AAA", "ZZZ"),
        "ZZZ": ("ZZZ", "ZZZ"),
    }


@pytest.mark.parametrize(
    "input_fn", [inputs.simple, inputs.multi_step, inputs.parallel, inputs.full]
)
def test_first_task(input_fn):
    result, _, input = input_fn()
    assert main.first_task(input) == result


@pytest.mark.parametrize(
    "input_fn", [inputs.simple, inputs.multi_step, inputs.parallel, inputs.full]
)
def test_second_task(input_fn):
    _, result, input = input_fn()
    assert main.second_task(input) == result
