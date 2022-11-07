#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : 
#           Luis Quintero | luisqtr.com
#           Michal Gnacek | gnacek.com
# Created Date: 2022/04
# =============================================================================
"""
Classes to help loading DRAP in a structured way
"""
# =============================================================================
# Imports
# =============================================================================

from .load_data import load_data_with_event_matching

# =============================================================================
# Classes
# =============================================================================

class Participant:
    
    sm_file_name = "slow_movement"
    fm_file_name = "fast_movement"
    video_1_file_name = "video_1"
    video_2_file_name = "video_2"
    video_3_file_name = "video_3"
    video_4_file_name = "video_4"
    video_5_file_name = "video_5"
    
    skip_sm = False
    skip_fm = False
    skip_video = False
    skip_video_1 = False
    skip_video_2 = False
    skip_video_3 = False
    skip_video_4 = False
    skip_video_5 = False
    
    def __init__(self, name, data_directory):
        self.name = name
        self.file_path = data_directory + "/" + name
        
    def getSlowMovementData(self):
        slowMovementFilesPath = self.file_path + "/" + self.sm_file_name
        return load_data_with_event_matching(slowMovementFilesPath + ".csv", True, slowMovementFilesPath + ".json")
    
    def getFastMovementData(self):
        fastMovementFilesPath = self.file_path + "/" + self.fm_file_name
        return load_data_with_event_matching(fastMovementFilesPath + ".csv", True, fastMovementFilesPath + ".json")
    
    def getVideo1Data(self):
        video_1_FilesPath = self.file_path + "/" + self.video_1_file_name
        return load_data_with_event_matching(video_1_FilesPath + ".csv", True, video_1_FilesPath + ".json")
    
    def getVideo2Data(self):
        video_2_FilesPath = self.file_path + "/" + self.video_2_file_name
        return load_data_with_event_matching(video_2_FilesPath + ".csv", True, video_2_FilesPath + ".json")
    
    def getVideo3Data(self):
        video_3_FilesPath = self.file_path + "/" + self.video_3_file_name
        return load_data_with_event_matching(video_3_FilesPath + ".csv", True, video_3_FilesPath + ".json")
    
    def getVideo4Data(self):
        video_4_FilesPath = self.file_path + "/" + self.video_4_file_name
        return load_data_with_event_matching(video_4_FilesPath + ".csv", True, video_4_FilesPath + ".json")
    
    def getVideo5Data(self):
        video_5_FilesPath = self.file_path + "/" + self.video_5_file_name
        return load_data_with_event_matching(video_5_FilesPath + ".csv", True, video_5_FilesPath + ".json")


class Video:

    def __init__(self, events, name):
        self.events = events
        self.name = name
        self.arousal_ratings = []
        self.valence_ratings = []
        for event in events:
          if("Valence:" in event):
                self.arousal_ratings.append(int(event.split("Arousal:",1)[1][0]))
                self.valence_ratings.append(int(event.split("Valence:",1)[1][0]))
        self.average_arousal = self.__average_rating(self.arousal_ratings)
        self.average_valence = self.__average_rating(self.valence_ratings)
                
    def __average_rating(self, ratings_list):
        sum = 0
        if(ratings_list==[]):
            return 0
        else:
            for rating in ratings_list:
                sum = sum + rating
            return round(sum/len(ratings_list),3)
                    
    def __average_arousal(self):
        return  self.__average_rating(self.arousal_ratings)
    
    def __average_valence(self):
        return self.__average_rating(self.valence_ratings)
    