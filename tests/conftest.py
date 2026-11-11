import pytest

from twinlens.core.gpt2 import GPT2

PROMPT = "The Colosseum is located in the city of"


@pytest.fixture(scope="session")
def model() -> GPT2:
    return GPT2.load("gpt2")
