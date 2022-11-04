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
from utils.enums import EmgVars, EmgMuscles, SessionSegment

# Import scientific 
import pandas as pd

import utils

# =============================================================================
# Main
# =============================================================================       


def GetColnameEmg(emgvar:EmgVars, muscle:EmgMuscles):
    """
    Returns a string with the column name for Emg data
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

def GetColnamesBasicsNonEmg():
    """
    Returns a list of colnames with essential non-EMG data
    """
    DATA_HEADSER_NON_EMG_BASICS = [
                                    #"Time",    # Included by default in `load_single_csv_data()`
                                    #"Frame",
                                    "Faceplate/FaceState","Faceplate/FitState",
                                    "HeartRate/Average","Ppg/Raw.ppg", 
                                    "Ppg/Raw.proximity",
                                    "Accelerometer/Raw.x","Accelerometer/Raw.y","Accelerometer/Raw.z",
                                    "Magnetometer/Raw.x","Magnetometer/Raw.y","Magnetometer/Raw.z",
                                    "Gyroscope/Raw.x","Gyroscope/Raw.y","Gyroscope/Raw.z",
                                    "Pressure/Raw"
                                ]

    return DATA_HEADSER_NON_EMG_BASICS


class Manager():

    """Columns found on the CSV file after converting with DabTools"""
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

    

    # Structure of the dataset containing the data.
    # The values of the dict correspond to filename where data is stored
    EXPERIMENT_SESSIONS_DICT = { str(session) : "" for session in SessionSegment }

    PROCESSED_EVENTS_DICT = {
            "Session": [],
            "Timestamp":[],
            "Event":[],
        }

    # Structure of the filepaths per user
    PARTICIPANT_DATA_DICT = {
        "folderid": "",         # Name of the folder
        "events": None,         # Events from all experiment segments are in a single file
        "segments": None,         # Timestamps for the beginning of the experiment segments
        "emotions": None,       # Subjective emotional data from all segments are in a single file
        "data": deepcopy(EXPERIMENT_SESSIONS_DICT),  # Data is stored per experiment session (>50MB/each file)
    }

    ### CONSTANTS
    DATA_FILE_EXTENSION = ".csv"
    EVENTS_FILE_EXTENSION = ".json"
    CONVERSION_TIMESTAMP_FROM_J2000_TO_UNIX = +946684800000 # in miliseconds

    # OUTPUT VALUES
    EVENTS_EXPERIMENT_FILENAME = "compiled_experimental_events.csv"
    SEGMENT_TIMESTAMPS_EXPERIMENT_FILENAME = "compiled_segment_timestamps.csv"
    EMOTIONS_SUBJECTIVE_FILENAME = "compiled_subjective_emotions.csv"
    JSON_INDEX_FILENAME = "data_tree_index.json"

    # MAIN VARIABLES TO ACCESS DATA

    # Filenames
    _folder_data_path = ""    # Root folder of the original dataset
    _index_file_path = ""     # Filepath for the json file containing the index

    # Debug
    _verbose = False

    # Data Variables
    index = None            # Dictionary with the dataset's index
    events = None           # Dictionary of Pandas DataFrame with Events
    segments = None         # Dictionary of Pandas DataFrame with Timestamps of each segment
    emotions = None         # Dictionary of Pandas DataFrame with Subjective Emotions
    data = None             # Dictionary of Pandas DataFrame with Emteq Data

    def __init__(self, folder_path, verbose=False):
        """
        This class loads the data from the remote video experiment.
        
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
            - data[0]["folderid"] > Returns the id of the participant's data
            - data[0]["events"] > Returns a pandas dataframe with the experimental events
                                    compiled among all the session segments.
            - data[0]["segments"] > Returns a pandas DataFrame indicating the start of each of the
                                    experimental stages (Videos: Negative, Neutral, Positive) or specific `videoId`
            - data[0]["emotions"] > Returns the subjective emotional values as reported
                                    by the participant, and stored in the `events.json`
            - data[0]["data"]["session"] > Returns a `string` indicating where the data
                        is located for the user `0` and the session `session`. The data needs
                        to be loaded individually because each file >50MB (~8GB in total)

        `session` is either str from the Enum `SessionSegment` or the string of the experiment session segment: 
                    ["fast_movement", "slow_movement", "video_1", 
                    "video_2", "video_3", "video_4", "video_5"]

        :param folder_path: Relative path to folder with data
        :type folder_path: str
        """
        
        self._folder_data_path = folder_path
        self._index_file_path = os.path.join(self._folder_data_path, self.JSON_INDEX_FILENAME)
        self._verbose = verbose

        self.load_or_create_index()
        return

    def load_or_create_index(self):
        """
        Analyzes the folder to see which files are available.
        Enables access to the variable `self.index`, which contains a 
        dictionary with path to key event and data files.
        It also creates the json file at the root of the dataset.

        :return: Nothing
        :rtype: None
        """

        # Entry condition
        if(self.__load_index_file() is not None):
            print("Index already exists: Loading from ", self._index_file_path)
            return
        
        ##### Create index from the dataset folder
        print("There is no index yet! Creating it in ", self._index_file_path)
    
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
                    files_index[counter_idx]["folderid"] = directory.name.split("_")[1]

                    if(self._verbose): print(f"\nDirectory >> {directory.name}")

                    # Store all the events in a new single .csv file
                    compiled_events = pd.DataFrame( deepcopy(self.PROCESSED_EVENTS_DICT) )
                    
                    # Scan participant's dir for specific files
                    with os.scandir(os.path.join(self._folder_data_path, directory.name)) as it2:
                        for file in it2:
                            
                            ## The session is defined by the filename (without extension)
                            session_name = file.name.split(".")[0]

                            if(file.name.endswith(self.EVENTS_FILE_EXTENSION)):
                                # File is an EVENT. Read it right away

                                if(self._verbose): print(f"\t Event>> {session_name}")

                                this_event_df = self.__load_single_event_file_into_pandas(os.path.join(self._folder_data_path, 
                                                                                            directory.name, 
                                                                                            file.name), 
                                                                                            session_name)

                                compiled_events = pd.concat([compiled_events, this_event_df], ignore_index=True)

                            elif (file.name.endswith(self.DATA_FILE_EXTENSION) and (session_name in self.EXPERIMENT_SESSIONS_DICT.keys()) ):
                                # File is DATA, too large, just store the path.
                                if(self._verbose):  print(f"\t Data>> {session_name}")
                                files_index[counter_idx]["data"][session_name] = os.path.join(directory.name, file.name)

                    # Separate in two files the experimental events and valence/arousal ratings
                    complete_experiment_events, experimental_segments, subjective_affect_ratings = self.__separate_exp_stages_and_emotion_ratings(compiled_events)

                    # Save the .csv files
                    filepath_temp = os.path.join(self._folder_data_path, directory.name, self.EVENTS_EXPERIMENT_FILENAME)
                    complete_experiment_events.to_csv(filepath_temp, index=True)
                    filepath_temp = os.path.join(self._folder_data_path, directory.name, self.SEGMENT_TIMESTAMPS_EXPERIMENT_FILENAME)
                    experimental_segments.to_csv(filepath_temp, index=True)
                    filepath_temp = os.path.join(self._folder_data_path, directory.name, self.EMOTIONS_SUBJECTIVE_FILENAME)
                    subjective_affect_ratings.to_csv(filepath_temp, index=True)

                    # Add to the index the separate files.
                    files_index[counter_idx]["events"] = os.path.join(directory.name, self.EVENTS_EXPERIMENT_FILENAME)
                    files_index[counter_idx]["segments"] = os.path.join(directory.name, self.SEGMENT_TIMESTAMPS_EXPERIMENT_FILENAME)
                    files_index[counter_idx]["emotions"] = os.path.join(directory.name, self.EMOTIONS_SUBJECTIVE_FILENAME)

                    print(f"\t Events compiled in {filepath_temp}")

                    # Prepare for next data
                    counter_idx = counter_idx + 1

        print(f"A total of {counter_idx} folders were found in the dataset")

        # Store the files in a JSON
        utils.create_json(files_index, self._index_file_path)

        print(f"Json file with index of the dataset was saved in {self._index_file_path}")

        # Global variable for the index
        self.index = files_index.copy()
        return

    def __load_index_file(self):
        """
        Loads the dictionary with the index file into memory.
        If error, returns None
        """
        try:  
            self.index = utils.load_json(self._index_file_path)
            self.index = { int(k):v for k,v in self.index.items() }
            return 0
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
            print("There is no index file loaded, loading index file...")
            self.load_or_create_index()
        else:
            ### Load events in dictionary
            self.events = {}
            for id, evt_path in self.index.items():
                # Iterate over participants
                self.events[id] = pd.read_csv(os.path.join(self._folder_data_path, evt_path["events"]), index_col="Time")
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
            print("There is no index file loaded, loading index file...")
            self.load_or_create_index()
        else:
            ### Load events in dictionary
            self.segments = {}
            for id, evt_path in self.index.items():
                # Iterate over participants
                self.segments[id] = pd.read_csv(os.path.join(self._folder_data_path, evt_path["segments"]), index_col="Time")
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
            print("There is no index file loaded, loading index file...")
            self.load_or_create_index()
        else:
            ### Load events in dictionary
            self.emotions = {}
            for id, evt_path in self.index.items():
                # Iterate over participants
                self.emotions[id] = pd.read_csv(os.path.join(self._folder_data_path, evt_path["emotions"]), index_col="Time")
                self.emotions[id].drop_duplicates(keep="first", inplace=True)
        return

    def __normalize_from_metadata():
        # TODO! # 
        pass

    def __load_single_csv_data(self, path_to_csv:str, 
                                    columns:list=None, 
                                    filter_wrong_timestamps:bool=True,
                                    apply_reference_timestamp_J2000:bool = True,
                                    filter_duplicates:bool = True,
                                    normalize_from_metadata:bool=True):
        """
        Filepath to CSV file to load.

        :param path_to_csv: Full path to CSV file to be loaded
        :param columns: Subset of columns to extract. You may use `EmgPathMuscles` to generate the list
        :param filter_wrong_timestamps: Remove the rows that contain timestamps <0 and >last_timestamp_in_file
        :param apply_reference_timestamp_J2000: Convert the timestamps from Unix to J2000 using metadata "#Time/Seconds.referenceOffset"
        :param filter_duplicates: Removes the duplicate rows from the pandas DataFrame, exclusing the timestamps in the index.

        :return: Data and metadata
        :rtype: A tuple with two pandas.DataFrames
        """

        # Metadata with the character '#'
        metadata = pd.read_csv( path_to_csv, sep=",", engine="c", on_bad_lines='skip', header=0, names = ["metadata","value"])
        metadata.set_index("metadata", inplace=True)
        
        # All lines that do not start with the character '#', therefore `comment="#"`
        data = pd.read_csv( path_to_csv, sep=",", comment="#", engine="c", header=0, names=self.DATA_HEADER_CSV)

        # Subselect some columns
        if columns is not None:
            data = data[ ["Time"] + columns ]

        # Filter data with invalid timestamps
        if(filter_wrong_timestamps):
            # Some timestamps carry over wrong timestamps due to high-freq data, 
            # thus remove samples with values greater than time in the last row
            data = data[ (data["Time"] < data["Time"].iloc[-1]) ]

        # Time as index in the DF
        data.set_index("Time", inplace=True)

        # Convert timestamps
        if (apply_reference_timestamp_J2000):
            # Extract from the metadata the J2000 reference value
            _ref_timestamp_J2000 = float(metadata["#Time/Seconds.referenceOffset"].value)
            data.index += (_ref_timestamp_J2000) # Transform from secs to msec

        # Convert values to corresponding units based on metadata
        if(normalize_from_metadata):
            data = self.__normalize_from_metadata(data, metadata)

        # Most data should contain duplicates if the raw data @2KHz are not chosen
        if filter_duplicates:
            data.drop_duplicates(keep="first", inplace=True)

        return data, metadata


    def load_data_from_participant(self, 
                                participant_idx:int, 
                                session_part:str, 
                                columns:list=None, 
                                use_J2000_timestamps:bool = False
                                ):
        """
        Loads the recorded data from a specific participant and a given 
        experiment session segment.
        
        :param participant_idx: Index of the participant (generally from 0 to 15)
        :param session_part: Unique key indicating which session segment to access. See `SessionSegment(Enum)`
        :param columns: List of columns to return from the dataset
        :param use_J2000_timestamps: Convert from Unix to J2000 format (useful to match with Event logs)
        :return: Tuple of two dataframes containing (data, metadata)
        :rtype: Tuple of two pandas DataFrames
        """
        path_to_requested_file = self.index[participant_idx]["data"][session_part]
        full_path_to_file = os.path.join(self._folder_data_path, path_to_requested_file)
        print("Loading from: ", full_path_to_file)

        return self.__load_single_csv_data(full_path_to_file, 
                                            columns = columns,
                                            apply_reference_timestamp_J2000 = use_J2000_timestamps)
    

    def __load_single_event_file_into_pandas(self, 
                        event_filepath, 
                        session_name,
                        convert_J2000_to_unix_seconds:bool = True):
        """
        Loads a file with events into a structured dictionary
        """
        dict_from_json = utils.load_json(event_filepath)
        
        # Transform to simpler dict compatible with Pandas
        organized_dict = deepcopy(self.PROCESSED_EVENTS_DICT)

        # Convert each key:value into an array with column names
        for event_info in dict_from_json:
            for k,v in event_info.items():
                organized_dict[k].append(v)

        # Repeat the session name as much as needed. It facilitates filtering
        organized_dict["Session"] = [session_name] * len(organized_dict["Timestamp"])

        # Create dataframe
        df = pd.DataFrame(deepcopy(organized_dict.copy()))

        # Convert from J2000 (in miliseconds) to Unix (in seconds)
        if(convert_J2000_to_unix_seconds):
            df["Timestamp"] = ( df["Timestamp"] + self.CONVERSION_TIMESTAMP_FROM_J2000_TO_UNIX ) / 1e3

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
        all_non_affect_events.index.rename("Time", inplace=True)
        
        timestamped_start_end_segments = self.__process_long_events_to_extract_experimental_segments(all_non_affect_events)

        #################################
        ########### Subjective emotions are Valence/Arousal ratings
        #################################
        subjective_affect_data = df[ QUERY_FILTER ]

        # Change index
        subjective_affect_data.set_index("Timestamp", inplace=True)
        subjective_affect_data.index.rename("Time", inplace=True)

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
        To be used as class labels for the TS
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
                                    video_label_negative: "VideoNegative",
                                    video_label_neutral: "VideoNeutral",
                                    video_label_positive: "VideoPositive",
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
        # tstamps_total_segments.reset_index(inplace=True) ## Do not uncomment, loading scripts always look for "Time" as index

        return tstamps_total_segments


############################
#### ENTRY POINT
############################

import sys, argparse

def help():
    m = f"""
        Experiment execution with dataset ''
        Parameters:
            
        """
    # print(m)
    return m

def main(args):
    input_folder_path = args.datasetroot 
    print(f"Analyzing folder {input_folder_path}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--datasetroot", type=str, required=True, help=f"Path to the dataset EMTEQ")

    # args = parser.parse_args()
    # main(args)

    print(" >>>> TESTING MANUALLY")
    data_loader_etl2 = Manager(os.path.join(THIS_PATH,"../data/"))

