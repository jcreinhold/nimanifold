#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nimanifold.data.sample.step

generate samples from windows of each image for a dataset

Author: Jacob Reinhold (jcreinhold@gmail.com)

Created on: Apr. 15, 2021
"""

__all__ = [
    'create_step_grid',
    'step_locs',
    'step_patches',
]

from typing import *

import numpy as np
from skimage.util import view_as_windows

from nimanifold.types import *
from nimanifold.data.sample.util import (
    create_grid,
    middle_slices,
    _middle_loc
)


def create_step_grid(shape: Shape, window: int, step: int) -> Grid:
    x, y, z = create_grid(shape)
    x = view_as_windows(x, window, step=step).reshape(-1, window, window, window)
    y = view_as_windows(y, window, step=step).reshape(-1, window, window, window)
    z = view_as_windows(z, window, step=step).reshape(-1, window, window, window)
    return x, y, z


def step_patches(img: Array, window: int = 40, step: Optional[int] = None, threshold: float = 0.,
                 **kwargs) -> Tuple[List[Array], List[int]]:
    if step is None:
        step = window
    patches, idxs = [], []
    windows = view_as_windows(img, window, step=step).reshape(-1, window, window, window)
    for i, w in enumerate(windows):
        if w.sum() > threshold:
            idxs.append(i)
            patches.append(w)
    return patches, idxs


def step_locs(grid: Grid, idxs: Optional[List[int]] = None) -> List[Array]:
    if idxs is not None:
        grid = [g[idxs] for g in grid]
    return [np.array(_middle_loc(xyz)) for xyz in zip(*grid)]


def _step_data_locs_slices(img: Array, grid: Grid, **kwargs) -> DataLocSlice:
    patches, idxs = step_patches(img, **kwargs)
    samples = [p.flatten() for p in patches]
    locs = step_locs(grid, idxs)
    slices = middle_slices(patches)
    return samples, locs, slices
