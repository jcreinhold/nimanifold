#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nimanifold.data.sample.util

general utilities for sampling a dataset

Author: Jacob Reinhold (jcreinhold@gmail.com)

Created on: Apr. 15, 2021
"""

__all__ = [
    'create_grid',
    'middle',
    'middle_slices',
    'project_dataset_to_sphere',
    'project_to_sphere'
]

from typing import *

from matplotlib import cm
import numpy as np

from nimanifold.types import *


def create_grid(shape: Shape) -> Grid:
    x = np.linspace(0, 1, shape[1])
    y = np.linspace(0, 1, shape[0])
    z = np.linspace(0, 1, shape[2])
    return np.meshgrid(x, y, z)


def middle_slices(patches: List[Array], axis: int = 2, n_rot: int = 3) -> List[Array]:
    shape = patches[0].shape
    if axis == 0:
        _slice = lambda x: np.rot90(x[shape[0] // 2, :, :], n_rot)
    elif axis == 1:
        _slice = lambda x: np.rot90(x[:, shape[1] // 2, :], n_rot)
    elif axis == 2:
        _slice = lambda x: np.rot90(x[:, :, shape[2] // 2], n_rot)
    else:
        raise ValueError(f'axis {axis} invalid. needs to be one of 0, 1, 2.')
    slices = [_slice(p) for p in patches]
    return slices


def middle(x: Array) -> Number:
    idx = tuple(np.asarray(x.shape) // 2)
    return x[idx]


def _middle_loc(xyz: SubGrid) -> Loc:
    x, y, z = xyz
    return (middle(x), middle(y), middle(z))


def project_to_sphere(x: Array) -> Array:
    norm = np.linalg.norm(x)
    if norm == 0.:
        raise ValueError('Norm of sample needs to be greater than zero.')
    return x / np.linalg.norm(x)


def project_dataset_to_sphere(x: Array, axis: int = -1) -> Array:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    if np.any(norm == 0.):
        raise ValueError('Norm of all samples need to be greater than zero.')
    return x / np.linalg.norm(x)


def _get_cmap(data: Array, cmap: str = 'Spectral') -> Array:
    return cm.get_cmap(cmap, len(np.unique(data)))(data)[:, :3]
