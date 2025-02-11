# type: ignore

import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from ctnorm.Robustness.sybil.model import Sybil
from ctnorm.Robustness.sybil.serie import Serie
from ctnorm.Robustness.sybil.utils.visualization import visualize_attentions, collate_attentions
import ctnorm.Robustness.sybil.utils.logging_utils

__all__ = ["Sybil", "Serie", "visualize_attentions", "collate_attentions", "__version__"]
