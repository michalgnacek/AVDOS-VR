#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luis-eduardo@dsv.su.se | luisqtr.com
# Created Date: 2022/11/01
# =============================================================================
"""
Configuration file for dataset DRAP

"""
# =============================================================================
# Imports
# =============================================================================

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

# =============================================================================
# GENERAL SETUP
# =============================================================================

# MAIN FOLDERS FOR OUTPUT FILES
ROOT = "./"                 # Root folder for all files respect to the notebook: Use "../" to run IPython, or "./" to run from VSCode

DATASET_FOLDER = ROOT + "datasets/"  # Main directory for datasets

SHOW_PLOTS = True           # Flag to avoid time-consuming plots that are already generated
EXPORT_PLOTS = True         # Flag to generate files of the plots. Requires SHOW_PLOTS=True

PLOT_FOLDER = ROOT+"plots/" # Main folder for plots
IMG_FORMAT = ".png"         # Format to export the plots

TEMP_FOLDER = ROOT+"temp/"  # Main folder for temp files with intermediate calculations
TEMP_FORMAT = ".npy"     # Extension for temp files created with pickle

RESULTS_FOLDER = ROOT+"results/"

# CSV files for cross-similarity matrices for each possible pair {Data Rep, Distance Measure}
PREFIX_DATASET = "dataset_"

# WORKFLOW MANAGEMENT REGARDING ITERATIONS TO LOAD OR GENERATE FILES
RELOAD_TRIES = 2            # Each step tries to create and load input_files maximum RELOAD_TRIES number of times

##########################
### DATASETS
##########################

# Input for the notebook
DATASET_MAIN =  "DRAP" 

# Original datasets
DATASET_ROOT_FOLDER = DATASET_FOLDER + DATASET_MAIN + "/"

##########################
### CLASSIFIERS
##########################

# Classes: Which column from the demographics.csv is used as target class label
CLASS_COLUMN_NAME = "videoId" #"user"  # "videoId": Tries to classify the videos. "user" tries to classify the people.

#### FEATURE-BASED CLASSIFIERS CLASSIFIERS SETUP

MC_RANDOM_SEED = 1234
CV_NUM_FOLDS = 10

# KNN
KNN_N_NEIGH = 9
# DT
DT_MAX_DEPTH = 100
# RF
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
# GBM
GBM_N_ESTIMATORS = 50
GBM_MAX_DEPTH = 5


#### STATE-OF-THE-ART CLASSIFIERS SETUP

# KNN-TS
KNN_TS_N_NEIGH = Classifiers.KNN_1
KNN_TS_DTW_WARPING_WINDOW = 0.05

# Mr-SEQL (Multivariate)
# No params required

# STSF (Univariate)
STSF_N_ESTIMATORS = 200

# TDE (Multivariate)
TDE_MAX_TIME = 5
TDE_MAX_ENSEMBLE_SIZE = 50
TDE_MAX_SELECTED_PARAMS = 50

# ROCKET (Multivariate)
ROCKET_N_KERNELS = 10000

# MiniRocket (Multivariate)
MINIROCKET_N_KERNELS = 10000
MINIROCKET_MAX_DILATIONS = 32
