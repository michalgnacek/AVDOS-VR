# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:10:33 2021

@author: Michal
"""
from load_data import load_data

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
        return load_data(slowMovementFilesPath + ".csv", True, slowMovementFilesPath + ".json")
    
    def getFastMovementData(self):
        fastMovementFilesPath = self.file_path + "/" + self.fm_file_name
        return load_data(fastMovementFilesPath + ".csv", True, fastMovementFilesPath + ".json")
    
    def getVideo1Data(self):
        video_1_FilesPath = self.file_path + "/" + self.video_1_file_name
        return load_data(video_1_FilesPath + ".csv", True, video_1_FilesPath + ".json")
    
    def getVideo2Data(self):
        video_2_FilesPath = self.file_path + "/" + self.video_2_file_name
        return load_data(video_2_FilesPath + ".csv", True, video_2_FilesPath + ".json")
    
    def getVideo3Data(self):
        video_3_FilesPath = self.file_path + "/" + self.video_3_file_name
        return load_data(video_3_FilesPath + ".csv", True, video_3_FilesPath + ".json")
    
    def getVideo4Data(self):
        video_4_FilesPath = self.file_path + "/" + self.video_4_file_name
        return load_data(video_4_FilesPath + ".csv", True, video_4_FilesPath + ".json")
    
    def getVideo5Data(self):
        video_5_FilesPath = self.file_path + "/" + self.video_5_file_name
        return load_data(video_5_FilesPath + ".csv", True, video_5_FilesPath + ".json")
    