#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nimanifold.data.csv

functions to load and process csv files

Author: Jacob Reinhold (jcreinhold@gmail.com)

Created on: Apr. 15, 2021
"""

__all__ = [
    'iacl_csv',
    'get_contrast_map',
    'get_patient_id_map',
    'get_site_map'
]

from typing import *

import pandas as pd

from nimanifold.types import *

CALABRESI_CSV = "/iacl/pg20/jacobr/calabresi/scripts/all_valid_v2.csv"
IXI_CSV = "/iacl/pg20/jacobr/ixi/norm/ixi.csv"


def iacl_csv(dataset: str, subtype: Optional[str] = None, site: Optional[str] = None) -> DataFrame:
    if dataset == 'calabresi':
        csv = pd.read_csv(CALABRESI_CSV)
        csv.rename(columns={'flair': 'filename', 'subject': 'id'}, inplace=True)
        if subtype is not None:
            csv = csv.query(f"type == '{subtype}'")
    elif dataset == 'ixi':
        csv = pd.read_csv()
        if subtype is not None:
            csv = csv.query(f"contrast == '{subtype}'")
        if site is not None:
            csv = csv.query(f"site == '{site}'")
    else:
        raise ValueError(f'dataset {dataset} not valid.')
    if csv.empty:
        raise ValueError(f'subtype {subtype} or site {site} not valid.')
    return csv


def get_attr_map(csv: DataFrame, attr: str, error: bool = False) -> dict:
    if hasattr(csv, attr):
        x = getattr(csv, attr).unique()
        out = {k: v for k, v in zip(x, range(len(x)))}
    else:
        if not error:
            out = None
        else:
            raise ValueError(f'Attribute {attr} must exist in the provided CSV file.')
    return out


def get_patient_id_map(csv: DataFrame) -> dict:
    return get_attr_map(csv, 'id', True)


def get_site_map(csv: DataFrame) -> dict:
    return get_attr_map(csv, 'site', False)


def get_contrast_map(csv: DataFrame) -> dict:
    return get_attr_map(csv, 'contrast', False)
