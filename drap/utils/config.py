#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : 
#           Luis Quintero | luisqtr.com
#           Michal Gnacek | gnacek.com
# Created Date: 2022/04
# =============================================================================
"""
Constants to help loading the DRAP dataset
"""
# =============================================================================
# Imports
# =============================================================================


# =============================================================================
# Constants
# =============================================================================

"""Columns found on the CSV file after converting with DabTools"""
TIME_COLNAME = "Time"

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

"""Basic set of non-EMG related columns obtained from a participant's reading"""
DATA_HEADSER_NON_EMG_BASICS = [
                                #"Time",    # Included by default when loading participant's data `load_single_csv_data()`
                                #"Frame",
                                "Faceplate/FaceState","Faceplate/FitState",
                                "HeartRate/Average","Ppg/Raw.ppg", 
                                "Ppg/Raw.proximity",
                                "Accelerometer/Raw.x","Accelerometer/Raw.y","Accelerometer/Raw.z",
                                "Magnetometer/Raw.x","Magnetometer/Raw.y","Magnetometer/Raw.z",
                                "Gyroscope/Raw.x","Gyroscope/Raw.y","Gyroscope/Raw.z",
                                "Pressure/Raw"
                            ]

# File extensions to look for when loading data
DATA_FILE_EXTENSION = ".csv"
EVENTS_FILE_EXTENSION = ".json"

# Offset to convert timestamps
CONVERSION_TIMESTAMP_FROM_J2000_TO_UNIX = +946684800000 # in miliseconds
