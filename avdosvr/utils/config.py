#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : 
#           Luis Quintero | luisqtr.com
#           Michal Gnacek | gnacek.com
# Created Date: 2022/04
# =============================================================================
"""
Constants to help loading the dataset
"""
# =============================================================================
# Imports
# =============================================================================


# =============================================================================
# Constants
# =============================================================================

"""Columns found on the CSV file after converting with DabTools"""

TIME_COLNAME = "Time"

# Columns in the order how they are found in the CSV files
DATA_HEADER_CSV = [
            "Frame","Time","Faceplate/FaceState","Faceplate/FitState",
            "Emg/ContactStates[RightFrontalis]","Emg/Contact[RightFrontalis]","Emg/Raw[RightFrontalis]","Emg/RawLift[RightFrontalis]","Emg/Filtered[RightFrontalis]","Emg/Amplitude[RightFrontalis]",
            "Emg/ContactStates[RightZygomaticus]","Emg/Contact[RightZygomaticus]","Emg/Raw[RightZygomaticus]","Emg/RawLift[RightZygomaticus]","Emg/Filtered[RightZygomaticus]","Emg/Amplitude[RightZygomaticus]",
            "Emg/ContactStates[RightOrbicularis]","Emg/Contact[RightOrbicularis]","Emg/Raw[RightOrbicularis]","Emg/RawLift[RightOrbicularis]","Emg/Filtered[RightOrbicularis]","Emg/Amplitude[RightOrbicularis]",
            "Emg/ContactStates[CenterCorrugator]","Emg/Contact[CenterCorrugator]","Emg/Raw[CenterCorrugator]","Emg/RawLift[CenterCorrugator]","Emg/Filtered[CenterCorrugator]","Emg/Amplitude[CenterCorrugator]",
            "Emg/ContactStates[LeftOrbicularis]","Emg/Contact[LeftOrbicularis]","Emg/Raw[LeftOrbicularis]","Emg/RawLift[LeftOrbicularis]","Emg/Filtered[LeftOrbicularis]","Emg/Amplitude[LeftOrbicularis]",
            "Emg/ContactStates[LeftZygomaticus]","Emg/Contact[LeftZygomaticus]","Emg/Raw[LeftZygomaticus]","Emg/RawLift[LeftZygomaticus]","Emg/Filtered[LeftZygomaticus]","Emg/Amplitude[LeftZygomaticus]",
            "Emg/ContactStates[LeftFrontalis]","Emg/Contact[LeftFrontalis]","Emg/Raw[LeftFrontalis]","Emg/RawLift[LeftFrontalis]","Emg/Filtered[LeftFrontalis]","Emg/Amplitude[LeftFrontalis]",
            "HeartRate/Average","Ppg/Raw.ppg","Ppg/Raw.proximity",
            "Accelerometer/Raw.x","Accelerometer/Raw.y","Accelerometer/Raw.z",
            "Magnetometer/Raw.x","Magnetometer/Raw.y","Magnetometer/Raw.z",
            "Gyroscope/Raw.x","Gyroscope/Raw.y","Gyroscope/Raw.z",
            "Pressure/Raw"
            ]

# Subset of columns used for normalization of data values to its corresponding units
EMG_SIGNAL_COLNAMES = [ a for a in DATA_HEADER_CSV if (("Emg/Raw[" in a) or ("Emg/Filtered[" in a) or ("Emg/Amplitude[" in a) )] # Do NOT apply to '/RawLift['
EMG_CONTACT_COLNAMES = [ a for a in DATA_HEADER_CSV if "Emg/Contact[" in a] # Do NOT apply to '/ContactStates['

FACEPLATE_COLNAMES = [ a for a in DATA_HEADER_CSV if "Faceplate/" in a]
HR_COLNAME = [ a for a in DATA_HEADER_CSV if "HeartRate/" in a]
PPG_COLNAMES = [ a for a in DATA_HEADER_CSV if "Ppg/" in a]

ACC_COLNAMES = [ a for a in DATA_HEADER_CSV if "Accelerometer/" in a]
MAG_COLNAMES = [ a for a in DATA_HEADER_CSV if "Magnetometer/" in a]
GYR_COLNAMES = [ a for a in DATA_HEADER_CSV if "Gyroscope/" in a]

"""Basic set of non-EMG related columns obtained from a participant's reading"""
DATA_NON_EMG_RECOMMENDED = [
                        #"Time",    # Included by default when loading participant's data `load_single_csv_data()`
                        #"Frame",
                        "Faceplate/FaceState","Faceplate/FitState",
                        "HeartRate/Average",
                        "Ppg/Raw.ppg", "Ppg/Raw.proximity"] \
                        + ACC_COLNAMES \
                        + MAG_COLNAMES \
                        + GYR_COLNAMES \
                        # + ["Pressure/Raw"]

EMOTION_RATINGS_COLNAMES = ["Valence","Arousal","RawX","RawY"]

# File extensions to look for when loading data
DATA_FILE_EXTENSION = ".csv"
EVENTS_FILE_EXTENSION = ".json"

# Offset to convert time timestamps
CONVERSION_TIMESTAMP_FROM_J2000_TO_UNIX = +946684800000 # in miliseconds
