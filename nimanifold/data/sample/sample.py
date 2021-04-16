#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nimanifold.data.sample.sample

generate samples either through creating
windows or randomly sampling for a dataset

Author: Jacob Reinhold (jcreinhold@gmail.com)

Created on: Apr. 15, 2021
"""

__all__ = [
    'get_samples'
]

from typing import *

from functools import partial

import nibabel as nib
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

from nimanifold.types import *
from nimanifold.data.csv import *
from nimanifold.data.sample.random import (
    _random_data_locs_slices,
)
from nimanifold.data.sample.step import (
    _step_data_locs_slices,
    create_step_grid
)
from nimanifold.data.sample.util import (
    _get_cmap,
    create_grid,
    project_dataset_to_sphere
)


def get_samples(csv: DataFrame,
                window: int = 40,
                step: Optional[int] = None,
                n_samples: Optional[int] = None,
                threshold: Optional[float] = None,
                to_sphere: bool = False,
                random: bool = False,
                progress: bool = True) -> Sample:
    patient_id_map = get_patient_id_map(csv)
    site_map = get_site_map(csv)
    contrast_map = get_contrast_map(csv)
    has_site = site_map is not None
    has_contrast = contrast_map is not None
    grid_creator = create_grid if random else \
        partial(create_step_grid, window=window, step=step)
    if step is None:
        step = window
    if threshold is None:
        threshold = float(window) / 4.
    sample_kwargs = dict(n_samples=n_samples,
                         step=step,
                         threshold=threshold,
                         window=window)
    sampler = partial(_random_data_locs_slices, **sample_kwargs) if random else \
        partial(_step_data_locs_slices, **sample_kwargs)
    data, locs, pids, slices, sites, contrasts = [], [], [], [], [], []
    grids = {}
    rows = enumerate(csv.iterrows())
    if progress:
        rows = tqdm(rows, total=csv.shape[0])
    for i, (_, row) in rows:
        fn = row.filename
        pid = row.id
        img = nib.load(fn).get_fdata()
        shape = img.shape
        if shape not in grids:
            grid = grid_creator(shape)
            grids = {shape: grid}
        data_, locs_, slices_ = sampler(img, grids[shape])
        N = len(data_)
        data.append(np.asarray(data_))
        locs.append(np.asarray(locs_))
        slices.append(np.asarray(slices_))
        pids.append(np.asarray([patient_id_map[pid]] * N))
        if has_site:
            site = row.site
            sites.append(np.asarray([site_map[site]] * N))
        if has_contrast:
            contrast = row.contrast
            contrasts.append(np.asarray([contrast_map[contrast]] * N))
    data = np.vstack(data)
    data, idxs = np.unique(data, axis=0, return_index=True)
    locs = np.vstack(locs)[idxs]
    locs = (locs - locs.min()) / (locs.max() - locs.min())
    slices = np.vstack(slices)[idxs]
    pids = np.concatenate(pids)[idxs]
    pids = _get_cmap(pids, 'gist_ncar')
    sites = _get_cmap(np.concatenate(sites)[idxs]) if has_site else None
    contrasts = _get_cmap(np.concatenate(contrasts)[idxs]) if has_contrast else None
    if to_sphere:
        data = project_dataset_to_sphere(data)
    else:
        data = preprocessing.scale(data)
    samples = Sample(data, locs, pids, slices, sites, contrasts)
    return samples
