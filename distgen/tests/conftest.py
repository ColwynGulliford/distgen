import pathlib

TESTS_PATH = pathlib.Path(__file__).resolve().parent
DISTGEN_PATH = TESTS_PATH.parent
REPO_ROOT = DISTGEN_PATH.parent
DOCS_PATH = REPO_ROOT / "docs"
EXAMPLES_PATH = DOCS_PATH / "examples"
EXAMPLES_DATA_PATH = EXAMPLES_PATH / "data"

assert EXAMPLES_PATH.exists()
