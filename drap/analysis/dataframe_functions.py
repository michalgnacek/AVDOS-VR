#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luis-eduardo@dsv.su.se | luisqtr.com
# Created Date: 2022/11/01
# =============================================================================
"""
Preprocessing DRAP dataset
"""
# =============================================================================
# Imports
# =============================================================================

import pandas as pd

# =============================================================================
# Main
# =============================================================================

def resample_dataframe(df, sampling_frequency_hz = 50):
    """
    Takes a dataframe with index as time in seconds and returns
    a resampled dataframe with the specific `sampling_frequency_hz` and
    with time index restarted from 0 seconds.

    The resampling happens repeating in `forward` direction first. Then a backwards fill
    is done to fill up the first values that are often empty because there
    are no affective ratings at the beginning of the sesssions.
    """
    
    df.index = pd.to_datetime(df.index, unit="s")
    df_resampled = df.resample(str(1/sampling_frequency_hz)+'S', origin='start').ffill()
    # The valence, arousal, rawX, rawY will contain null values before the first value is captured. Fill with first value.
    df_resampled = df_resampled.fillna(method="backfill")
    # Put the data back to 0 seconds
    df_resampled.index -= df_resampled.index[0]
    # Transform from datetime to float
    df_resampled.index = df_resampled.index.total_seconds()
    return df_resampled