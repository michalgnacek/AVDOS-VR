# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:51:23 2021

@author: Michal
"""

import json
from enum import Enum

def read_jsonfile(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data    

def message_not_found(message):
     print("Message not found: ", message)
     
def message_found(message):
     print("Message found: ", message)
     
class FileType(Enum):
    JSON = 1
    RAW = 2
    CSV = 3
    OTHER = 4
    
def getFileType(file_name):
        if (file_name.endswith('.raw')):
            return FileType.RAW
        elif (file_name.endswith('.csv')):
            return FileType.CSV
        elif (file_name.endswith('.json')):
            return FileType.JSON
        else:
            print("Unknown file type found:" + file_name)
            return FileType.OTHER

def drop_start_frames(data, frames_to_drop):
    return data.drop(data.index[0:frames_to_drop])