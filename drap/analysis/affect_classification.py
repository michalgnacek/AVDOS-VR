#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luis-eduardo@dsv.su.se | luisqtr.com
# Created Date: 2022/11/01
# =============================================================================
"""
Affect classification
"""
# =============================================================================
# Imports
# =============================================================================

import os
THIS_PATH = str(os.path.dirname(os.path.abspath(__file__)))

from enum import IntEnum, unique
import numpy as np
import pandas as pd

# =============================================================================
# Enums with variables for each step of the analysis
# =============================================================================

@unique
class Classifiers(IntEnum):
    HIVE_COTEv1 = -1
    STSF = -2
    ROCKET = -3
    MiniRocket = -4
    MrSEQL = -5
    TDE = -6

    KNN = 0     # Used in general, not tied to num of neighbors
    KNN_1 = 1
    KNN_7 = 7   # IMT has 5 classes
    KNN_9 = 9   # UCR UWaveGesture has 8 classes
    KNN_11 = 11 # Tsinghua has 9 classes
    DT = 20   # Decision Tree
    RF = 21   # Random Forest
    GBM = 22  # Gradient Boosting Machine
    
    # Add more classifiers here



# =============================================================================
# Functions
# =============================================================================

def filter_outliers_from_df(df, num_std = 5):
    """
    Takes a multidimensional dataFrame and filter the values
    that are `num_std` standard deviations away from the mean value
    of the column.

    First, it transforms the value in np.nan. Then, it imputes the
    value with backward filling, and then with forward filling, in case
    the missing values are generated on the extremes of the time-series.

    Returns the filtered dataset
    """
    mask = (( df > (df.mean() + num_std*df.std())) | ( df < (df.mean() - num_std*df.std())) )
    df[ mask ] = np.nan
    df_filtered = df.fillna(method="backfill", axis=0)
    df_filtered = df_filtered.fillna(method="ffill", axis=0)
    print(f"\tTotal NAs --> Generated={df.isna().sum().sum()} - After imputation={df_filtered.isna().sum().sum()}")
    return df_filtered
    
def calculate_statistical_features(df):
    """
    Calculates the following features per column in the dataframe,
    adding a suffix for each processed column:
        - mean:     mean
        - std:      standard devaition
        - min:      minimum value
        - max:      maximum value
        - median:   median
        - irq:      interquartile range
        - pnv:      proportion of negative values
        - ppv:      proportion of positive values
        - skew:     skewness of the distribution
        - kurt:     kurtosis of the distribution
        - energy:   sum of squared absolute values
    """

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    FUNCTIONS_FEATURES = {
        "mean":     np.mean,
        "std":      np.std,
        "min":      np.min,
        "max":      np.max,
        "median":   np.median,
        "irq":      stats.iqr,
        "pnv":      (lambda y: y[y<0].size/y.size),
        "ppv":      (lambda y: y[y>0].size/y.size),
        "skew":     stats.skew,
        "kurt":     stats.kurtosis,
        "energy":   (lambda y: np.sum(np.abs(y)**2) ),
    }
    
    # Store results with features per columns
    df_features_results = { }

    for feat_name,feat_func in FUNCTIONS_FEATURES.items():
        for col_name in list(df.columns):
            df_features_results[f"{col_name}_{feat_name}"] = [ feat_func(df[col_name]) ]

    return pd.DataFrame(df_features_results)
