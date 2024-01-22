import pytest


@pytest.mark.parametrize(
    "text,result",
    [
        (
            r"versifi\u00e9, poli, inclusif, avec un l\u00e9ger grain de po\u00e9sie",
            "versifié, poli, inclusif, avec un léger grain de poésie",
        ),
        (
            "avec un grain de philosophie \nstyle Yoda bougon",
            "avec un grain de philosophie style Yoda bougon",
        ),
    ],
)
def test_clean_text(text, result):
    from llm import clean_text

    assert clean_text(text) == result
