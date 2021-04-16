#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nimanifold.plot.generic

generic plotting routines

Author: Jacob Reinhold (jcreinhold@gmail.com)

Created on: Apr. 15, 2021
"""

__all__ = [
    'plot',
    'scatter_imgs'
]

from typing import *

from matplotlib import offsetbox
import matplotlib.pyplot as plt
import numpy as np

from nimanifold.types import *

TICK_PARAMS = dict(
    left=False,
    bottom=False,
    labelleft=False,
    labelbottom=False
)


def plot(data: Array,
         colors: Optional[Array] = None,
         ax: Axes = None,
         title: str = None,
         slices: Array = None) -> None:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(data[:, 0], data[:, 1], c=colors, s=3.)
    ax.set_facecolor("black")
    ax.axis('scaled')
    ax.xaxis.set_tick_params(**TICK_PARAMS)
    ax.yaxis.set_tick_params(**TICK_PARAMS)
    if slices is not None:
        scatter_imgs(data, slices, ax)
    if title is not None:
        plt.title(title)


def scatter_imgs(data: Array, slices: Array, ax: Axes):
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(data.shape[0]):
            dist = np.sum((data[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [data[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(slices[i], cmap=plt.cm.gray_r),
                data[i])
            ax.add_artist(imagebox)
