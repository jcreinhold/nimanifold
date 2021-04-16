#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nimanifold.types

nimanifold-specific types

Author: Jacob Reinhold (jcreinhold@gmail.com)

Created on: Apr. 15, 2021
"""

__all__ = [
    'Array',
    'Axes',
    'DataFrame',
    'DataLocSlice',
    'Grid',
    'Loc',
    'Number',
    'Sample',
    'Shape',
    'SubGrid'
]

from typing import *

from copy import copy

from matplotlib.axes import Axes
import numpy as np
import pandas as pd

Array = np.ndarray
DataFrame = pd.DataFrame
DataLocSlice = Tuple[List[Array], List[Array], List[Array]]
Grid = Tuple[Array, Array, Array]
Loc = Tuple[int, int, int]
Number = Union[int, float]
Shape = Tuple[int, int, int]
SubGrid = Tuple[Array, Array, Array]


class Sample:
    def __init__(self,
                 data: Array,
                 locs: Array,
                 pids: Array,
                 slices: Array,
                 sites: Optional[Array] = None,
                 contrasts: Optional[Array] = None):
        self.data = data
        self.locs = locs
        self.pids = pids
        self.slices = slices
        self.sites = sites
        self.contrasts = contrasts
        self.is_valid()

    def __len__(self):
        return self.data[0]

    def __repr__(self):
        return f"{len(self)} Samples"

    def is_valid(self):
        N = len(self)
        assert (self.locs.shape[0] == N)
        assert (self.locs.shape[1] == 3)
        assert (self.pids.shape[0] == N)
        assert (self.pids.shape[1] == 3)
        assert (self.slices.shape[0] == N)
        if self.sites is not None:
            assert (self.sites.shape[0] == N)
            assert (self.sites.shape[1] == 3)
        if self.contrasts is not None:
            assert (self.contrasts.shape[0] == N)
            assert (self.contrasts.shape[1] == 3)

    def new_data(self, data: Array):
        sample = copy(self)
        sample.data = data
        sample.is_valid()
        return sample

    def subsample(self, n: int):
        N = len(self)
        assert (n <= N)
        idxs = np.random.choice(N, size=n, replace=False)
        sample = Sample(
            self.data[idxs],
            self.locs[idxs],
            self.pids[idxs],
            self.slices[idxs],
            self.sites[idxs] if self.sites is not None else None,
            self.contrasts[idxs] if self.contrasts is not None else None,
        )
        return sample
