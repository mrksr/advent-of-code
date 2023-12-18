import inputs
import main
import pytest


@pytest.mark.parametrize(
    "input,result", [inputs.simple, inputs.multi_step, inputs.full]
)
def test_first_task(input, result):
    assert main.first_task(input) == result
