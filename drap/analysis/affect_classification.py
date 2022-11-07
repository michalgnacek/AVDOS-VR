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
