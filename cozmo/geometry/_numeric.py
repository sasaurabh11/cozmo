"""Small numeric helpers shared by the geometry stages."""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np


@contextmanager
def quiet_fp():
    """Silence spurious floating-point flags from float32 matmul.

    numpy on Apple's Accelerate raises divide/overflow/invalid flags on float32
    matmul even when every result is finite (verified on the sample capture:
    1.59M points, zero non-finite). Silencing them here keeps real warnings
    visible instead of drowned in noise.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            yield
