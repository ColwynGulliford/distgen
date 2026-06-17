#!/usr/bin/env python
import h5py
import pytest

from .. import Generator
from .conftest import EXAMPLES_DATA_PATH


@pytest.fixture(scope="module")
def generator() -> Generator:
    G = Generator(str(EXAMPLES_DATA_PATH / "gaussian.in.yaml"))
    G["n_particle"] = 2000
    G.run()
    return G


def _assert_roundtrip(original: Generator, reloaded: Generator) -> None:
    # Input is fully restored
    assert reloaded.input == original.input

    # Particles are restored and re-runnable
    reloaded.run()
    assert len(reloaded.particles["x"]) == len(original.particles["x"])
    assert reloaded["start:type"] == original["start:type"]


def test_archive_roundtrip_to_filename(generator: Generator, tmp_path) -> None:
    """Archive to a filename (internal h5py.File), reload, and re-run."""
    fname = str(tmp_path / "archive.h5")
    generator.archive(fname)

    # Correct particle layout: particles/<species>, not particles/particles/<species>
    with h5py.File(fname, "r") as f:
        assert "input" in f
        assert "particles" in f
        assert "particles" not in f["particles"]

    reloaded = Generator()
    reloaded.load_archive(fname)
    _assert_roundtrip(generator, reloaded)


def test_archive_roundtrip_into_group(generator: Generator, tmp_path) -> None:
    """Archive into an existing subgroup (the lume-astra pattern), reload, and re-run."""
    fname = str(tmp_path / "archive_group.h5")
    with h5py.File(fname, "w") as f:
        generator.archive(f.create_group("distgen"))

    reloaded = Generator()
    reloaded.load_archive(fname)
    _assert_roundtrip(generator, reloaded)


def test_archive_skips_none_input_values(generator: Generator, tmp_path) -> None:
    """None-valued input keys are dropped on write and treated as absent on read."""
    fname = str(tmp_path / "archive_none.h5")
    generator.archive(fname)

    with h5py.File(fname, "r") as f:
        assert not any(v is None for v in f["input"].attrs.values())


def test_load_archive_without_input_group(generator: Generator, tmp_path) -> None:
    """Legacy archives lacking an 'input' group still load without error."""
    fname = str(tmp_path / "legacy.h5")
    generator.archive(fname)

    # Remove the input group to emulate a legacy archive
    with h5py.File(fname, "a") as f:
        del f["input"]

    reloaded = Generator()
    reloaded.load_archive(fname)
    assert len(reloaded.particles["x"]) == len(generator.particles["x"])
