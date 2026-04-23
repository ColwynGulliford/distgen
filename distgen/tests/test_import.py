import distgen
import pytest


def test_importable():
    assert hasattr(distgen, "__version__")
    assert isinstance(distgen.__version__, str)


@pytest.mark.parametrize("name", distgen.__all__)
def test_lazy_imports(name):

    getattr(distgen, name)
