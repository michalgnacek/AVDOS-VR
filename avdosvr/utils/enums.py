#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : 
#           Luis Quintero | luisqtr.com
#           Michal Gnacek | gnacek.com
# Created Date: 2022/04
# =============================================================================
"""
Enums help loading the dataset in a structured way
"""
# =============================================================================
# Imports
# =============================================================================

from enum import Enum

# =============================================================================
# Enums
# =============================================================================

class EmgVars(Enum):
    ContactStates = "ContactStates"
    Contact = "Contact"
    Raw = "Raw"
    RawLift = "RawLift"
    Filtered = "Filtered"
    Amplitude = "Amplitude"
    def __str__(self):
        return super().value.__str__()
    

class EmgMuscles(Enum):
    """
    Enum to access the Muscles from EMG
    """
    RightFrontalis = "RightFrontalis"
    RightZygomaticus = "RightZygomaticus"
    RightOrbicularis = "RightOrbicularis"
    CenterCorrugator = "CenterCorrugator"
    LeftOrbicularis = "LeftOrbicularis"
    LeftZygomaticus = "LeftZygomaticus"
    LeftFrontalis = "LeftFrontalis"
    def __str__(self):
        return super().value.__str__()
    

class EventMessages(Enum):
    """
    Short messages to find in the json files to
    filter the corresponding events.
    """
    SLOW_MOVEMENT_SIGNAL_CHECK_MESSAGE = "Start of signal check. Started data recording. Slow_movement_SignalCheck"
    SLOW_MOVEMENT_FINISHED_MESSAGE = "Slow movement study: finished data recording"

    FAST_MOVEMENT_SIGNAL_CHECK_MESSAGE = "Start of signal check. Started data recording. Fast_movement_SignalCheck"
    FAST_MOVEMENT_FINISHED_MESSAGE = "Fast movement study: finished data recording"

    VIDEO_SIGNAL_CHECK_MESSAGE = "Start of signal check. Started data recording. Video_SignalCheck"
    VIDEO_PLAYING_REST_VIDEO = "Playing rest video"
    VIDEO_FINISHED_PLAYING_REST_VIDEO = "Finished playing rest video"
    VIDEO_CATEGORY_FINISHED = "Video category finished"
    VIDEO_FINISHED_PLAYING_ALL_VIDEOS = "Finished playing all videos"
    VIDEO_FINISHED_STUDY = "Video ratings study: finished data recording"
    def __str__(self):
        return super().value.__str__()


class SessionSegment(Enum):
    """
    Enum to access the dictionary with the data per video
    """
    # FastMovement = "fast_movement" # Deleted from public dataset
    # SlowMovement = "slow_movement" # Deleted from public dataset
    video1 = "video_1"
    video2 = "video_2"
    video3 = "video_3"
    video4 = "video_4"
    video5 = "video_5"
    def __str__(self):
        return super().value.__str__()

class AffectSegments(Enum):
    VideosPositive = "Positive"
    VideosNegative = "Negative"
    VideosNeutral = "Neutral"
    def __str__(self):
        return super().value.__str__()