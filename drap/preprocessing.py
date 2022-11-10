#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luisqtr.com
# Created Date: 2022/04
# =============================================================================
"""
Functions to load the datasets from Emteq Labs regarding behavioral analysis
in VR environments.

The functions loads dataset collected from the EmteqPRO mask, and code is based
on the scripts from: https://github.com/emteqlabs/demo-analysis-scripts

The specifications for the CSV files are available from Emteq Support: 
https://support.emteqlabs.com/data/CSV.html 
"""
# =============================================================================
# Imports
# =============================================================================

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
THIS_PATH = str(os.path.dirname(os.path.abspath(__file__)))

# Import data manipulation libraries
from copy import deepcopy

import utils.files_handler
import utils.load_data
from utils import config
from utils.enums import EmgVars, EmgMuscles, SessionSegment, AffectSegments

# Import scientific 
import numpy as np
import pandas as pd

import utils

# =============================================================================
# Main
# =============================================================================       

# Functions to generate colnames for the dataset

def GetColnameEmg(emgvar:EmgVars, muscle:EmgMuscles):
    """
    Returns a string with the column name for Emg data
    based on type of variable `emgvar` and facial `muscle`
    """
    return "Emg/" + str(emgvar) + "[" + str(muscle) + "]"

def GetColnamesFromEmgVariableType(emgvar:EmgVars):
    """
    Returns a list with all column names for the 
    EMG type `emgvar`
    """
    colnames = []
    for muscle in EmgMuscles:
        colnames.append( GetColnameEmg(emgvar, muscle) )
    return colnames

def GetColnamesFromEmgMuscle(muscle:EmgMuscles):
    """
    Returns a list with all column Emg variable types
    for the EMG muscle `muscle`
    """
    colnames = []
    for emg_dtype in EmgVars:
        colnames.append( GetColnameEmg(emg_dtype, muscle) )
    return colnames

## Shortcut to access DRAP colnames

COLNAMES_EMG_RAW = GetColnamesFromEmgVariableType(EmgVars.Raw)
COLNAMES_EMG_FILTERED = GetColnamesFromEmgVariableType(EmgVars.Filtered)
COLNAMES_EMG_AMPLITUDE = GetColnamesFromEmgVariableType(EmgVars.Amplitude)
COLNAMES_EMG_CONTACT = GetColnamesFromEmgVariableType(EmgVars.Contact)
COLNAMES_EMG_CONTACT_STATES = GetColnamesFromEmgVariableType(EmgVars.ContactStates)

COLNAMES_FACEPLATE = config.FACEPLATE_COLNAMES
COLNAMES_HR = config.HR_COLNAME
COLNAMES_PPG = config.PPG_COLNAMES

COLNAMES_ACCELEROMETER = config.ACC_COLNAMES
COLNAMES_MAGNETOMETER = config.MAG_COLNAMES
COLNAMES_GYROSCOPE = config.GYR_COLNAMES

COLNAMES_RECOMMENDED = COLNAMES_EMG_AMPLITUDE + config.DATA_NON_EMG_RECOMMENDED


############################
#### MAIN CLASS TO GENERATE INDEX
############################

class Manager():  

    # Structure of the dataset containing the data.
    # The values of the dict correspond to filename where data is stored
    EXPERIMENT_SESSIONS_DICT = { str(segment) : "" for segment in SessionSegment }

    PROCESSED_EVENTS_DICT = {
            "Session": [],
            "Timestamp":[],
            "Event":[],
        }

    # Structure of the filepaths per user
    PARTICIPANT_DATA_DICT = {
        "participant_id": "",         # Name of the folder
        "protocol": "",         # Whether is v1 or v2
        "events": None,         # Events from all experiment segments are in a single file
        "segments": None,         # Timestamps for the beginning of the experiment segments
        "emotions": None,       # Subjective emotional data from all segments are in a single file
        "data": deepcopy(EXPERIMENT_SESSIONS_DICT),  # Data is stored per experiment session (>50MB/each file)
    }

    # OUTPUT VALUES
    EVENTS_EXPERIMENT_FILENAME = "compiled_experimental_events.csv"
    SEGMENT_TIMESTAMPS_EXPERIMENT_FILENAME = "compiled_protocol_segment.csv"
    EMOTIONS_SUBJECTIVE_FILENAME = "compiled_emotion_ratings.csv"
    JSON_INDEX_FILENAME = "drap_tree_index.json"
    SUMMARY_DF_FILENAME = "summary_data.csv"

    # MAIN VARIABLES TO ACCESS DATA

    # Filenames
    _folder_data_path = ""    # Root folder of the original dataset
    _index_file_path = ""     # Filepath for the json file containing the index

    # Debug
    _verbose = False

    # Data Variables
    index = None            # Dictionary with the dataset's index
    summary = None          # Pandas DataFrame summarizing the dataset
    events = None           # Dictionary of Pandas DataFrame with Events
    segments = None         # Dictionary of Pandas DataFrame with Timestamps of each segment
    emotions = None         # Dictionary of Pandas DataFrame with Subjective Emotions
    data = None             # Dictionary of Pandas DataFrame with Emteq Data

    def __init__(self, drap_folder_root:str, 
                        verbose=False, 
                        force_index_regeneration:bool=False, 
                        index_files_path:str = None):
        """
        Analyzes the folder to see which files are available.
        Enables access to the variable `self.index`, which contains a 
        dictionary with filepath to key `events`, `emotions` and `data`.

        It  creates the json file at the root of the `index_files_path`
        and individual .csv inside the participant's folder.
        
        A session consists of three segments: 
            1) slow movement, 2) fast movement, and 3) videos (1 to 5).

        The datasets with suffix "_v2" also contain an **expression calibration stage** where 
        the participants were asked to perform 3 repetitions of: `smile`, `frown`, 
        `surprise`. The calibration was performed at the start of the `slow_movement_segment`.
        
        The output of this class is a configured object that loads the list of available
        participants and their data. However, it does not load the whole dataset of the user
        because each dataset can be around 500MB. To load each participant's dataset
        use the method `load_data_from_participant()`.

        Download link for original dataset: Request to Emteq Labs (emteqlabs.com/)

        The output of the data loading process is:
            - data[0]["participant_id"] > Returns the id of the participant's data
            - data[0]["protocol"] > Returns the type of experimental protocol conducted
            - data[0]["events"]   > Returns the filepath for a pandas dataframe with the experimental events
                                    compiled among all the session segments.
            - data[0]["segments"] > Returns the filepath for a pandas DataFrame indicating the start of each of the
                                    experimental stages (Videos: Negative, Neutral, Positive) and specific `videoId`
            - data[0]["emotions"] > Returns the subjective emotional values as reported
                                    by the participant, and stored in the `events.json`
            - data[0]["data"]["segment"] > Returns a `string` indicating where the data
                        is located for the user `0` and the session `session`. The data needs
                        to be loaded individually because each file >50MB (~8GB in total)

        `session` is either str from the Enum `SessionSegment` or the string of the experiment session segment: 
                    ["fast_movement", "slow_movement", "video_1", 
                    "video_2", "video_3", "video_4", "video_5"]

        :param folder_path: Relative path to folder with data
        :type folder_path: str
        :param force_index_regeneration: Forces regeneration of index, even if it exists:
        :type force_index_regeneration: bool
        :param index_files_path: Folder where the temp files for the index will be stored. If None, they
                                are stored at the same level from the dataset in a folder called `temp/drap_index/`. 
        :type index_files_path: str
        :return: Dictionary
        :rtype: dict
        """
        
        # Define where main dataset is stored
        self._folder_data_path = drap_folder_root

        # Define where temporary index files will be stored
        _temp_folder_index_files = index_files_path if (index_files_path is not None) else os.path.join(self._folder_data_path,"../temp")
        _temp_folder_index_files = os.path.join(_temp_folder_index_files, "drap_index/")
        self._index_file_path = os.path.join(_temp_folder_index_files, self.JSON_INDEX_FILENAME)

        # Debug Verbosity
        self._verbose = verbose
        
        # Entry condition
        if force_index_regeneration:
            print("Forcing index construction!!", self._index_file_path)
            self.__generate_index(_temp_folder_index_files)
        else:
            if (not self.__load_index_file()):        
                # Create index from the dataset folder if it does not exist
                # otherwise it will automatically loaded in 
                print("There is no index yet! Creating it in ", self._index_file_path)
                self.__generate_index(_temp_folder_index_files)
            else:
                # Index file already exists
                print("Index already exists: Loading from ", self._index_file_path)

        # Load the summary, events, emotions, and segments. Not data because it is too large.
        self.load_event_files()
        self.load_emotion_files()
        self.load_segments_files()

        # Load or generate summary
        self.summary =  self.__generate_basic_summary_df()
        # Save summary file
        filepath_temp = os.path.join(_temp_folder_index_files, self.SUMMARY_DF_FILENAME)
        utils.files_handler.check_or_create_folder(filepath_temp)
        self.summary.to_csv(filepath_temp, index=False)
        return

    def __generate_index(self, index_folder:str):
        """
        Analyzes the data's folder and creates a file
        with the index
        """

        _temp_folder_index_files = index_folder

        ### GENERATING INDEX
        # Dictionary to store files
        files_index = {}

        # Look for zip files and extract all in the same directory
        counter_idx = 0
        with os.scandir(self._folder_data_path) as it:
            for directory in it:
                ### DIRECTORIES AS PARTICIPANTS
                if( not directory.name.startswith(".") and directory.is_dir() ):                    
                    # A folder is equivalent to a participant

                    # Add the participant data to the file index.
                    # The index is a sequential number from `counter_idx`
                    files_index[counter_idx] = deepcopy(self.PARTICIPANT_DATA_DICT)   # Empty dict for data
                    files_index[counter_idx]["participant_id"] = directory.name.split("_")[1]
                    files_index[counter_idx]["protocol"] = "v2" if ("v2" in directory.name) else "v1"

                    if(self._verbose): print(f"\nDirectory >> {directory.name}")

                    # Store all the events in a new single .csv file
                    compiled_events = pd.DataFrame( deepcopy(self.PROCESSED_EVENTS_DICT) )
                    
                    # Scan participant's dir for specific files
                    with os.scandir(os.path.join(self._folder_data_path, directory.name)) as it2:
                        for file in it2:
                            
                            ## The session is defined by the filename (without extension)
                            session_name = file.name.split(".")[0]

                            if(file.name.endswith(config.EVENTS_FILE_EXTENSION)):
                                # File is an EVENT. Read it right away

                                if(self._verbose): print(f"\t Event>> {session_name}")

                                this_event_df = self.__load_single_event_file_into_pandas(os.path.join(self._folder_data_path, 
                                                                                            directory.name, 
                                                                                            file.name), 
                                                                                            session_name)

                                compiled_events = pd.concat([compiled_events, this_event_df], ignore_index=True)

                            elif (file.name.endswith(config.DATA_FILE_EXTENSION) and (session_name in self.EXPERIMENT_SESSIONS_DICT.keys()) ):
                                # File is DATA, too large, just store the path.
                                if(self._verbose):  print(f"\t Data>> {session_name}")
                                files_index[counter_idx]["data"][session_name] = os.path.join(self._folder_data_path, directory.name, file.name)

                    # Separate in two files the experimental events and valence/arousal ratings
                    complete_experiment_events, experimental_segments, subjective_affect_ratings = self.__separate_exp_stages_and_emotion_ratings(compiled_events)

                    # Save the .csv files
                    filepath_temp = os.path.join(_temp_folder_index_files, directory.name, self.EVENTS_EXPERIMENT_FILENAME)
                    utils.files_handler.check_or_create_folder(filepath_temp)
                    complete_experiment_events.to_csv(filepath_temp, index=True)
                    filepath_temp = os.path.join(_temp_folder_index_files, directory.name, self.SEGMENT_TIMESTAMPS_EXPERIMENT_FILENAME)
                    experimental_segments.to_csv(filepath_temp, index=True)
                    filepath_temp = os.path.join(_temp_folder_index_files, directory.name, self.EMOTIONS_SUBJECTIVE_FILENAME)
                    subjective_affect_ratings.to_csv(filepath_temp, index=True)
                    if(self._verbose): print(f"\t Events compiled in {filepath_temp}")

                    # Add to the index the separate files.
                    files_index[counter_idx]["events"] =    os.path.join(_temp_folder_index_files,directory.name, self.EVENTS_EXPERIMENT_FILENAME)
                    files_index[counter_idx]["segments"] =  os.path.join(_temp_folder_index_files,directory.name, self.SEGMENT_TIMESTAMPS_EXPERIMENT_FILENAME)
                    files_index[counter_idx]["emotions"] =  os.path.join(_temp_folder_index_files,directory.name, self.EMOTIONS_SUBJECTIVE_FILENAME)

                    # Prepare for next data
                    counter_idx = counter_idx + 1

        print(f"A total of {counter_idx} folders were found in the dataset")

        # Store the files in a JSON
        utils.files_handler.create_json(files_index, self._index_file_path)

        print(f"Json file with index of the dataset was saved in {self._index_file_path}")

        self.index = files_index
        return True

    def __load_single_event_file_into_pandas(self, 
                        event_filepath, 
                        session_name,
                        convert_J2000_to_unix_seconds:bool = True):
        """
        Loads a file with events into a structured dictionary
        """
        dict_from_json = utils.files_handler.load_json(event_filepath)
        
        # Transform to simpler dict compatible with Pandas
        organized_dict = deepcopy(self.PROCESSED_EVENTS_DICT)

        # Convert each key:value into an array with column names
        for event_info in dict_from_json:
            for k,v in event_info.items():
                organized_dict[k].append(v)

        # Repeat the session name as much as needed. It facilitates filtering
        organized_dict["Session"] = [session_name] * len(organized_dict["Timestamp"])

        # Create dataframe
        df = pd.DataFrame( deepcopy(organized_dict.copy()) )

        # Convert from J2000 (in miliseconds) to Unix (in seconds)
        # All .json files containing events are originally in J2000 format
        if(convert_J2000_to_unix_seconds):
            df["Timestamp"] = (df["Timestamp"] + config.CONVERSION_TIMESTAMP_FROM_J2000_TO_UNIX)/1e3
        return df

    def __separate_exp_stages_and_emotion_ratings(self, compiled_events_dataframe:pd.DataFrame):
        """
        Takes the dataframe that compiles all the event files, and
        produces two dataframes: One contains the events related to the experiments
        and the other contains time-series values with valence and arousal.
        """
        df = compiled_events_dataframe

        # Criteria to know whether it's experimental or emotional data
        QUERY_FILTER = (df["Event"].str.startswith("Valence"))

        ########### All experimental stages in a single dataset
        all_non_affect_events = df[ ~QUERY_FILTER ]

        all_non_affect_events.set_index("Timestamp", inplace=True)
        all_non_affect_events.index.rename(config.TIME_COLNAME, inplace=True)
        
        timestamped_start_end_segments = self.__process_long_events_to_extract_experimental_segments(all_non_affect_events)

        #################################
        ########### Subjective emotions are Valence/Arousal ratings
        #################################
        subjective_affect_data = df[ QUERY_FILTER ]

        # Change index
        subjective_affect_data.set_index("Timestamp", inplace=True)
        subjective_affect_data.index.rename(config.TIME_COLNAME, inplace=True)

        # Extract the emotional data from the string. First splitting by "," and then by ":" every two values
        emotions = subjective_affect_data["Event"].str.split(",").apply(lambda x: [v.split(":")[1] for v in x])
        # The resulting frame is a Series of Lists. Transform into DataFrame
        emotions = pd.DataFrame(emotions.tolist(), 
                                index=subjective_affect_data.index, 
                                columns=["Valence","Arousal","RawX","RawY"])

        # Join with original timestamp and session segment.
        subjective_affect_data = subjective_affect_data.join(emotions)
        subjective_affect_data.drop(["Event"], axis=1, inplace=True)

        return all_non_affect_events, timestamped_start_end_segments, subjective_affect_data

    def __process_long_events_to_extract_experimental_segments(self, experimental_events):
        """
        Extract the starting time and end time of each of the experimental stages. 
        To be used to segment the TS between stages
        """

        VIDEO_ID_FOR_RESTING_VIDEO = -1
        KEYWORD_SEGMENT_BEGINNING = "Start"
        KEYWORD_SEGMENT_END = "End"

        # All experimental stages start with an Event saying "Playing ___"
        events_filter = experimental_events[experimental_events.Event.str.contains("Playing")]
        
        # Find the video file corresponding to each emotion. `video_2`, `video_3`, or `video_4`
        video_label_negative = events_filter[ events_filter.Event.str.contains("Category name: Negative") ].Session.values[0]
        video_label_positive = events_filter[ events_filter.Event.str.contains("Category name: Positive") ].Session.values[0]
        video_label_neutral = events_filter[ events_filter.Event.str.contains("Category name: Neutral") ].Session.values[0]

        # Dict to map video filename to emotion category
        map_session_to_emotion = {
                                    "fast_movement":"fast_movement",
                                    "slow_movement":"slow_movement",
                                    # Randomized order
                                    video_label_negative: str(AffectSegments.VideosNegative),
                                    video_label_neutral: str(AffectSegments.VideosNeutral),
                                    video_label_positive: str(AffectSegments.VideosPositive),
                                    # Test videos
                                    "video_1":"video_1",
                                    "video_5":"video_5",
                                }

        #### Find events related to the beginning of each video or rest stage.
        tstamps_start_videos = events_filter[events_filter.Event.str.startswith("Playing video number:")].copy()
        tstamps_start_videos["Segment"] = tstamps_start_videos["Session"].map(map_session_to_emotion) 
        tstamps_start_videos["VideoId"] = tstamps_start_videos.Event.str.split(":").apply( lambda x: int(x[1]))
        tstamps_start_videos["Trigger"] = KEYWORD_SEGMENT_BEGINNING

        tstamps_start_rest = events_filter [events_filter.Event.str.contains("Playing rest video")].copy()
        tstamps_start_rest["Segment"] = tstamps_start_rest["Session"].map(map_session_to_emotion)
        tstamps_start_rest["VideoId"] = VIDEO_ID_FOR_RESTING_VIDEO
        tstamps_start_rest["Trigger"] = KEYWORD_SEGMENT_BEGINNING

        ### Find events related to the end of each video, or rest stage
        events_filter_end = experimental_events[ experimental_events.Event.str.contains("Finished playing") ]

        tstamps_end_videos = events_filter_end [events_filter_end.Event.str.startswith("Finished playing video number:")].copy()
        tstamps_end_videos["Segment"] = tstamps_end_videos["Session"].map(map_session_to_emotion)
        tstamps_end_videos["VideoId"] = tstamps_end_videos.Event.str.split(":").apply( lambda x: int(x[1]))
        tstamps_end_videos["Trigger"] = KEYWORD_SEGMENT_END

        tstamps_end_rest = events_filter_end [events_filter_end.Event.str.startswith("Finished playing rest video")].copy()
        tstamps_end_rest["Segment"] = tstamps_end_rest["Session"].map(map_session_to_emotion)
        tstamps_end_rest["VideoId"] = VIDEO_ID_FOR_RESTING_VIDEO
        tstamps_end_rest["Trigger"] = KEYWORD_SEGMENT_END

        # Find the video Id playing per each group
        tstamps_total_segments = pd.concat([tstamps_start_videos,tstamps_start_rest,tstamps_end_videos,tstamps_end_rest])
        tstamps_total_segments.drop("Event", axis=1, inplace=True)
        tstamps_total_segments.sort_index(inplace=True)
        # tstamps_total_segments.reset_index(inplace=True) ## Do not uncomment, loading scripts always look for "Time" (config.DATA_HEADER_CSV) as index

        return tstamps_total_segments

    def __load_index_file(self):
        """
        Loads the dictionary with the index file into memory.
        The participant id is loaded as integer
        If error, returns None
        """
        try:  
            self.index = utils.files_handler.load_json(self._index_file_path)
            # Accessing the participants as numeric ids, not as strings
            self.index = { int(k):v for k,v in self.index.items() }
            return True
        except:
            return None

    def load_event_files(self):
        """
        Loads the dictionary containing the experimental events from each participant.
        Access all the events in a DataFrame from the participant 0 as:
            - dataset_loader.events[0]
        
        :return: Events during all the experiment
        :rtype: Pandas DataFrame
        """
        if self.index is None:
            print("There is no index file loaded, Create an index of the dataset...")
        else:
            ### Load events in dictionary
            self.events = {}
            for id, evt_path in self.index.items():
                # Iterate over participants
                self.events[id] = pd.read_csv(os.path.join(self._folder_data_path, evt_path["events"]), index_col=config.TIME_COLNAME)
        return

    def load_segments_files(self):
        """
        Loads the dictionary containing the experimental events from each participant.
        Access all the events in a DataFrame from the participant 0 as:
            - dataset_loader.segments[0]
        
        :return: Events during all the experiment
        :rtype: Pandas DataFrame
        """
        if self.index is None:
            print("There is no index file loaded, Create an index of the dataset...")
        else:
            ### Load events in dictionary
            self.segments = {}
            for id, evt_path in self.index.items():
                # Iterate over participants
                self.segments[id] = pd.read_csv(os.path.join(self._folder_data_path, evt_path["segments"]), index_col=config.TIME_COLNAME)
        return

    def load_emotion_files(self):
        """
        Loads the dictionary containing the self-reported emotional data from each participant.
        Access all the emotion data in a DataFrame from the participant 0 as:
            - dataset_loader.emotions[0]
        
        :return: Reported emotional values during all the experiment
        :rtype: Pandas DataFrame
        """
        if self.index is None:
            print("There is no index file loaded, Create an index of the dataset...")
        else:
            ### Load events in dictionary
            self.emotions = {}
            for id, evt_path in self.index.items():
                # Iterate over participants
                self.emotions[id] = pd.read_csv(os.path.join(self._folder_data_path, evt_path["emotions"]), index_col=config.TIME_COLNAME)
                self.emotions[id].drop_duplicates(keep="first", inplace=True)
        return

    def __generate_basic_summary_df(self):
        """
        Takes the index, events, emotions, and segments to create a compiled 
        dataframe that summarizes the data.
        This summary considers the whole file, without synchronizing the data
        with respect to the timestamps in the [Start,End] of each experimental segment.
        """
        df_sum = None
        for pid,pdata in self.index.items():
            print(f"Participant {pid} with folder id: {pdata['participant_id']} was part of protocol: {pdata['protocol']}")

            # Summary grouped per session segment    
            for segtype in self.events[pid]["Session"].unique():
                
                ######
                # Summary of events
                events_data = self.events[pid]
                Q = ( (events_data["Session"] == segtype) )
                event_filtered = events_data[Q].reset_index()

                ######
                # Summary of emotions
                emotions_data = self.emotions[pid]
                Q = ( (emotions_data["Session"] == segtype) )
                emotions_filtered = emotions_data[Q].reset_index()

                # Summarize dataframe
                current_summary = {
                            "index_id": pid,
                            "participant_id": pdata['participant_id'],
                            "protocol": pdata['protocol'],
                            "Segment":segtype,
                            "Events_N": event_filtered.shape[0],
                            "Events_duration": event_filtered["Time"].iloc[-1] - event_filtered["Time"].iloc[0],
                            "Emotions_N": emotions_filtered.shape[0],
                            "Emotions_duration": (emotions_filtered["Time"].iloc[-1] - emotions_filtered["Time"].iloc[0]) if emotions_filtered.shape[0]>0 else np.nan,
                            "Emotions_Valence_avg": emotions_filtered["Valence"].mean(),
                            "Emotions_Arousal_avg": emotions_filtered["Arousal"].mean()
                        }

                # Convert from values to list, to adapt to DataFrame
                current_summary = { k:[v] for k,v in current_summary.items() }
                current_summary = pd.DataFrame.from_dict(current_summary)

                df_sum = current_summary if (df_sum is None) else pd.concat([df_sum, current_summary], ignore_index=True)
        # Finished creating all files
        return df_sum

    def load_data_from_participant(self, 
                                participant_idx:int, 
                                session_segment:str, 
                                columns:list=None, 
                                convert_timestamps_to_unix:bool = True,
                                normalize_data_units = False,
                                ):
        """
        Loads the recorded data from a specific participant and a given 
        experiment session segment.
        
        :param participant_idx: Index of the participant (generally from 0 to 15)
        :param session_segment: Unique key indicating which session segment to access. See `SessionSegment(Enum)`
        :param columns: List of columns to return from the dataset
        :param convert_timestamps_to_unix: Convert data to Unix format (useful to match with Event logs)
        :param normalize_data_units: Apply normalization from metadata into the physio data
        :return: Tuple of two dataframes containing (data, metadata)
        :rtype: Tuple of two pandas DataFrames
        """
        path_to_requested_file = self.index[participant_idx]["data"][session_segment]
        full_path_to_file = os.path.join(self._folder_data_path, path_to_requested_file)
        print("Loading from: ", full_path_to_file)

        return utils.load_data.load_single_csv_data(full_path_to_file, 
                                            columns = columns,
                                            normalize_reference_time = convert_timestamps_to_unix,
                                            normalize_from_metadata=normalize_data_units)
    
    def obtain_order_experimental_segments(self, participant_idx:int):
        """
        Returns an array with the order of the experimental stage
        for the `participant_idx`
        """
        # Event sequence
        EVENT_TEXT_SEQUENCE = "Category sequence:"
        keys_containing_sync_event = self.events[participant_idx].Event.str.startswith(EVENT_TEXT_SEQUENCE)
        cat_sequence = self.events[participant_idx][ keys_containing_sync_event ].iloc[0] # Choose first event
        video_seq = cat_sequence.Event.split(":")[1].split(",")
        video_seq = [x.strip() for x in video_seq]
        return video_seq

    def calculate_info_from_segment(self, participant_idx:int, affective_segment:str):
        """
        Processes a dataframe of segments timestamps and returns a tuple with:
            - rest_tstamp_start: Timestamp when the resting video started
            - rest_tstamp_end: Timestamp when the resting video ended
            - video_tstamp_start: Timestamp when the corresponding video stage started
            - video_tstamp_end: Timestamp when the corresponding video stage ended
            - video_filename: Filename of the file corresponding to the affective stage
        
        :param participant_idx: Index of the participant (generally from 0 to 15)
        :param affective_segment: Unique key indicating which affective segment to access. See `AffectSegments(Enum)`
        :return: Tuple with five objects as described above.
        :rtype: Tuple
        """

        df_segments = self.segments[participant_idx]

        # Filter the segment corresponding to the intended video
        df_segments = df_segments[ df_segments["Segment"] == affective_segment]

        # Find the beginning and end of the RESTING (VideoId == -1)
        rest_start = df_segments[ (df_segments["Trigger"]=="Start") & (df_segments["VideoId"] == -1)].index.min()
        rest_end = df_segments[ (df_segments["Trigger"]=="End") & (df_segments["VideoId"] == -1)].index.max()

        # The segment watching the VIDEO (VideoId != -1)
        video_start = df_segments[ (df_segments["Trigger"]=="Start") & (df_segments["VideoId"] != -1)].index.min()
        video_end = df_segments[ (df_segments["Trigger"]=="End") & (df_segments["VideoId"] != -1)].index.max()

        # Which file should be loaded to access the required data
        video_filename = df_segments["Session"].iloc[0]

        # Correct the few situations when the video starts before resting ends for few miliseconds
        if rest_end > video_start:
            video_start = rest_end
        
        return (rest_start, rest_end, video_start, video_end, video_filename)

    def calculate_video_id_end_timestamps(self, participant_idx:int, affective_segment:str):
        """
        Returns a dataframe with the timestamp where each VideoID of the
        specific `affective_segment` **finish**.
        :param participant_idx: Index of the participant (generally from 0 to 15)
        :param affective_segment: Unique key indicating which affective segment to access. See `AffectSegments(Enum)`
        :return: Array denoting where each videoId finishes
        :rtype: pandas DataFrame
        """
        # EVENT_TEXT_SEQUENCE = "Finished playing video number:" # It will return when the event finished!
        # keys_containing_sync_event = df_events.Event.str.startswith(EVENT_TEXT_SEQUENCE)
        # videos_seq = df_events[ keys_containing_sync_event ] # Choose all video numbers
        # videos_ending = videos_seq.Event.str.split(":")
        # video_id_end_timestamp = videos_ending.apply((lambda x: int(x[1])))
        # video_id_end_timestamp = pd.DataFrame({"VideoID":video_id_end_timestamp})

        df_segments = self.segments[participant_idx]

        # Filter the segment corresponding to the intended video
        df_segments = df_segments[ df_segments["Segment"] == affective_segment]

        # Find the end of each video stage
        video_id_end_timestamp = df_segments[ (df_segments["Trigger"]=="End") ]
        video_id_end_timestamp = video_id_end_timestamp[ ["VideoId"] ]
        
        return video_id_end_timestamp

    def load_data_from_affect_segment(self,
                                participant_idx:int, 
                                affective_segment:str, 
                                columns:list=None, 
                                **kwargs
                                ):
        """
        Returns the data from the rest and video stages of a
        specific participant and affective segment 
        (`enums.AffectSegments`, or ["Positive","Neutral","Negative"]).
        By default, the data is loaded applying the normalization of the data 
        according to the units and accessing all the columns (columns=None).
        
        :param participant_idx: Index of the participant (generally from 0 to 15)
        :param affective_segment: Unique key indicating which affective segment to access. See `AffectSegments(Enum)`
        :param columns: List of columns to return from the dataset
        :param convert_timestamps_to_unix: Convert data to match timestamps in event logs)
        :param normalize_data_units: Apply normalization from metadata into the physio data
        :return: Tuple of two dataframes containing (data_rest, data_video)
        :rtype: Tuple of two pandas DataFrames
        """

        r_t0, r_t1, v_t0, v_t1, video_filename = self.calculate_info_from_segment(participant_idx, affective_segment)

        # Load video corresponding to desired Experimental stage
        data, metadata = self.load_data_from_participant(participant_idx = participant_idx, 
                                                                session_segment = video_filename,
                                                                normalize_data_units = True,
                                                                columns = columns, **kwargs)

        # Filter data between stages
        data_rest = data[ (data.index >= r_t0) & (data.index < r_t1) ]
        data_video = data[ (data.index >= v_t0) & (data.index < v_t1) ]

        return (data_rest, data_video)
    
    def load_emotions_from_affect_segment(self,
                                participant_idx:int, 
                                affective_segment:str
                                ):
        """
        Returns the emotions from the rest and video stages of a
        specific participant and affective segment 
        (`enums.AffectSegments`, or ["Positive","Neutral","Negative"]).

        :return: Tuple of two dataframes containing (data_rest, data_video)
        :rtype: Tuple of two pandas DataFrames
        """

        r_t0, r_t1, v_t0, v_t1, video_filename = self.calculate_info_from_segment(participant_idx, affective_segment)

        # Filter emotions between the ranges
        Q = (self.emotions[participant_idx].index >= r_t0) & \
                (self.emotions[participant_idx].index < r_t1 ) & \
                (self.emotions[participant_idx].Session == video_filename)
        emotions_rest = self.emotions[participant_idx][ Q ].drop("Session", axis=1)

        Q = (self.emotions[participant_idx].index >= v_t0) & \
                (self.emotions[participant_idx].index < v_t1 ) & \
                (self.emotions[participant_idx].Session == video_filename)
        emotions_video = self.emotions[participant_idx][ Q ].drop("Session", axis=1)

        return (emotions_rest, emotions_video)



############################
#### ENTRY POINT
############################

import sys, argparse

_FILE_DESCRIPTION = f"""
        This file generates an index for the dataset DRAP, to facilitate its analysis.
        The index is stored in the folder "temp/drap_index/"

        The file is easier used in a notebook. See example in `notebook/1_preprocess...ipynb`
        """

def main(args):
    if(args.dataset):
        print(" >>>> TESTING MANUALLY")
        Manager(os.path.join(THIS_PATH, args.dataset),
                verbose=args.verbose)
        print(f"Generating index for dataset in folder: {args.dataset}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DRAP dataset",
                                        description=_FILE_DESCRIPTION,
                                        epilog="See parameters running `python drap.preprocessing.py --help`")

    parser.add_argument("--dataset", type=str, required=True, help=f"Root path to the dataset DRAP")
    parser.add_argument("-v", "--verbose", action='store_true', help=f"Show verbose output")

    args = parser.parse_args()
    main(args)

