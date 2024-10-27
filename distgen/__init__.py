from .generator import Generator as Generator
from .physical_constants import PHYSICAL_CONSTANTS as PHYSICAL_CONSTANTS
# from . import _version

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"
