from freezegun import freeze_time
import pytest


@pytest.fixture
def gpt():
    from llm import AOFGPT

    return AOFGPT(
        api_key="fake_key",
        instructions="Nous sommes le $date et il est $time.",
        default_style="xxx",
    )


def test_la_date_est_dans_le_prompt(gpt):
    with freeze_time("2024-01-24 23:15:00"):
        prompt = gpt.system_prompt(context="xxx")
    assert "Nous sommes le 24 janvier 2024 et il est 23:15." in prompt
