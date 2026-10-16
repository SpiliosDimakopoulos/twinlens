import pytest

from twinlens.core.gpt2 import GPT2


@pytest.fixture(scope="session")
def model() -> GPT2:
    return GPT2.load("gpt2")
