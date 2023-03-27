# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:39:37 2022

@author: Michal Gnacek
"""

#%%
def skip_participant(participantObject):
    if(participantObject.name == "participant_360_v2"): #sm file 0 data
        participantObject.skip_sm = True
    if(participantObject.name == "participant_375_v2"): #video 5 file missing
        participantObject.skip_video_5 = True
        participantObject.skip_video_4 = True
        participantObject.skip_video_3 = True
        participantObject.skip_video_2 = True
        participantObject.skip_video_1 = True
        participantObject.skip_video = True
        participantObject.skip_fm = True
        participantObject.skip_sm = True
    if(participantObject.name == "participant_340_v2"): #poor fit
        participantObject.skip_video_5 = True
        participantObject.skip_video_4 = True
        participantObject.skip_video_3 = True
        participantObject.skip_video_2 = True
        participantObject.skip_video_1 = True
        participantObject.skip_video = True
        participantObject.skip_fm = True
        participantObject.skip_sm = True
    if(participantObject.name == "participant_246"): #poor fit
        participantObject.skip_video_5 = True
        participantObject.skip_video_4 = True
        participantObject.skip_video_3 = True
        participantObject.skip_video_2 = True
        participantObject.skip_video_1 = True
        participantObject.skip_video = True
        participantObject.skip_fm = True
        participantObject.skip_sm = True
    if(participantObject.name == "participant_313"): #high alexithymia
        participantObject.skip_video_5 = True
        participantObject.skip_video_4 = True
        participantObject.skip_video_3 = True
        participantObject.skip_video_2 = True
        participantObject.skip_video_1 = True
        participantObject.skip_video = True
        participantObject.skip_fm = True
        participantObject.skip_sm = True
    return participantObject