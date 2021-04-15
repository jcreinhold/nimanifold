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
    'Samples',
    'Shape',
    'SubGrid'
]

from typing import *

from matplotlib.axes import Axes
import numpy as np
import pandas as pd

Array = np.ndarray
DataFrame = pd.DataFrame
DataLocSlice = Tuple[List[Array], List[Array], List[Array]]
Grid = Tuple[Array, Array, Array]
Loc = Tuple[int, int, int]
Number = Union[int, float]
Samples = Tuple[Array, Array, Array, Array, Array, Array]
Shape = Tuple[int, int, int]
SubGrid = Tuple[Array, Array, Array]
