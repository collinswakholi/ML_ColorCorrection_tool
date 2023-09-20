import sys
from ColorCorrectionML.ColorCorrectionML import ColorCorrectionML
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = 'unknown'
    
__all__ = ['ColorCorrectionML']

if "pdoc" in sys.modules:
    with open("README.md", "r") as fh:
        _readme = fh.read()