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

import nibabel as nib
import numpy as np
from sklearn import preprocessing

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
                random: bool = False) -> Samples:
    patient_id_map = get_patient_id_map(csv)
    site_map = get_site_map(csv)
    contrast_map = get_contrast_map(csv)
    data, locs, pids, slices, sites, contrasts = [], [], [], [], [], []
    grids = {}
    if step is None:
        step = window
    if threshold is None:
        threshold = float(window) / 4.
    sample_kwargs = dict(n_samples=n_samples,
                         step=step,
                         threshold=threshold,
                         window=window)
    for i, (_, row) in enumerate(csv.iterrows()):
        fn = row.filename
        pid = row.id
        img = nib.load(fn).get_fdata()
        shape = img.shape
        if shape not in grids:
            if random:
                grid = create_grid(shape)
            else:
                grid = create_step_grid(shape, window, step)
            grids = {shape: grid}
        if random:
            samples, locs, slices = _random_data_locs_slices(img, grids[shape],
                                                             **sample_kwargs)
        else:
            samples, locs, slices = _step_data_locs_slices(img, grids[shape],
                                                           **sample_kwargs)
        N = len(samples)
        data.append(np.asarray(samples))
        locs.append(np.asarray(locs))
        slices.append(np.asarray(slices))
        pids.append(np.asarray([patient_id_map[pid]] * N))
        if site_map is not None:
            site = row.site
            sites.append(np.asarray([site_map[site]] * N))
        if contrast_map is not None:
            contrast = row.contrast
            contrasts.append(np.asarray([contrast_map[contrast]] * N))
    data = np.vstack(data)
    data, idxs = np.unique(data, axis=0, return_index=True)
    locs = np.vstack(locs)[idxs]
    locs = (locs - locs.min()) / (locs.max() - locs.min())
    pids = np.concatenate(pids)[idxs]
    pids = _get_cmap(pids, 'gist_ncar')
    if site_map is not None:
        sites = _get_cmap(np.concatenate(sites)[idxs])
    if contrast_map is not None:
        contrasts = _get_cmap(np.concatenate(contrasts)[idxs])
    if to_sphere:
        data = project_dataset_to_sphere(data)
    else:
        data = preprocessing.scale(data)
    return data, locs, pids, slices, sites, contrasts
