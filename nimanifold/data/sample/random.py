#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nimanifold.data.sample.random

generate samples from random crops for a dataset

Author: Jacob Reinhold (jcreinhold@gmail.com)

Created on: Apr. 15, 2021
"""

from typing import *
from nimanifold.types import *

import numpy as np

from nimanifold.data.sample.util import (
    middle_slices,
    _middle_loc
)


class Index:
    def __init__(self, i1: int, i2: int, j1: int, j2: int, k1: int, k2: int):
        self.i1, self.i2 = i1, i2
        self.j1, self.j2 = j1, j2
        self.k1, self.k2 = k1, k2

    def __call__(self, x: np.ndarray):
        return x[self.i1:self.i2, self.j1:self.j2, self.k1:self.k2]


class CropBase:
    def __init__(self,
                 out_dim: int,
                 output_size: Union[tuple, int, list],
                 n_samples: int = 1,
                 threshold: Optional[float] = None,
                 pct: Tuple[float, float] = (0., 1.),
                 axis: int = 0):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size,)
            for _ in range(out_dim - 1):
                self.output_size += (output_size,)
        else:
            assert len(output_size) == out_dim
            self.output_size = output_size
        self.out_dim = out_dim
        self.n_samples = n_samples
        self.thresh = threshold
        self.pct = pct
        self.axis = axis

    def _get_sample_idxs(self, img: Array) -> Loc:
        """ get the set of indices from which to sample (foreground) """
        # mask is a tuple of length 3
        mask = np.where(img >= (img.mean() if self.thresh is None else self.thresh))
        c = np.random.randint(0, len(mask[0]))  # choose the set of idxs to use
        h, w, d = [m[c] for m in mask]  # pull out the chosen idxs
        return h, w, d

    def _offset_by_pct(self, h: int, w: int, d: int) -> Tuple[Loc, Loc]:
        s = (h, w, d)
        hml = wml = dml = 0
        hmh = wmh = dmh = 0
        i0, i1 = int(s[self.axis] * self.pct[0]), int(s[self.axis] * (1. - self.pct[1]))
        if self.axis == 0:
            hml += i0
            hmh += i1
        elif self.axis == 1:
            wml += i0
            wmh += i1
        else:
            dml += i0
            dmh += i1
        return (hml, wml, dml), (hmh, wmh, dmh)

    def __repr__(self) -> str:
        s = '{name}(output_size={output_size}, threshold={thresh}, N={n_samples)'
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)


class RandomCrop3D(CropBase):
    """
    Generate N randomly-centered cropped samples from a 3D image
    Args:
        output_size (tuple or int): Desired output size.
            If int, cube crop is made.
    """

    def __init__(self,
                 output_size: Union[tuple, int, list],
                 n_samples: int = 1,
                 threshold: Optional[float] = None,
                 pct: Tuple[float, float] = (0., 1.),
                 axis: int = 0):
        super().__init__(3, output_size, n_samples, threshold, pct, axis)

    def __call__(self, img: Array) -> Tuple[List[Array], List[Index]]:
        *cs, h, w, d = img.shape
        hh, ww, dd = self.output_size
        (hml, wml, dml), (hmh, wmh, dmh) = self._offset_by_pct(h, w, d)
        max_idxs = (h - hmh - hh // 2, w - wmh - ww // 2, d - dmh - dd // 2)
        min_idxs = (hml + hh // 2, wml + ww // 2, dml + dd // 2)
        x = img[0] if len(cs) > 0 else img  # use the first image to determine sampling, if multimodal
        samples, idxs = [], []
        for _ in range(self.n_samples):
            s_idxs = self._get_sample_idxs(x)
            i, j, k = [i if min_i <= i <= max_i else max_i if i > max_i else min_i
                       for max_i, min_i, i in zip(max_idxs, min_idxs, s_idxs)]
            oh = 0 if hh % 2 == 0 else 1
            ow = 0 if ww % 2 == 0 else 1
            od = 0 if dd % 2 == 0 else 1
            i1, i2 = i - hh // 2, i + hh // 2 + oh
            j1, j2 = j - ww // 2, j + ww // 2 + ow
            k1, k2 = k - dd // 2, k + dd // 2 + od
            s = img[..., i1:i2, j1:j2, k1:k2]
            samples.append(s)
            idxs.append(Index(i1, i2, j1, j2, k1, k2))
        return samples, idxs


def random_patches(img: Array,
                   window: int = 40,
                   n_samples: int = 1,
                   threshold: float = 0.,
                   **kwargs) -> Tuple[List[Array], List[Index]]:
    cropper = RandomCrop3D(window, n_samples, threshold)
    patches, idxs = cropper(img)
    return patches, idxs


def random_locs(grid: Grid, idxs: List[Index]) -> List[Loc]:
    grids = [[idx(g) for g in grid] for idx in idxs]
    return [np.array(_middle_loc(xyz)) for xyz in grids]


def _random_data_locs_slices(img: Array, grid: Grid, **kwargs) -> DataLocSlice:
    patches, idxs = random_patches(img, **kwargs)
    samples = [p.flatten() for p in patches]
    locs = random_locs(grid, idxs)
    slices = middle_slices(patches)
    return samples, locs, slices
