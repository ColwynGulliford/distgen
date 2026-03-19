def test_importable():
    import distgen

    assert hasattr(distgen, "__version__")
    assert isinstance(distgen.__version__, str)
