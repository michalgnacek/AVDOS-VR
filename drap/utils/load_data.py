#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : 
#           Michal Gnacek | gnacek.com
#           Luis Quintero | luisqtr.com
# Created Date: 2021/01/08
# =============================================================================
"""
Utility functions to deal with files management
"""
# =============================================================================
# Imports
# =============================================================================

import pandas as pd
import numpy as np
import json
import re
from io import StringIO

from . import config

# =============================================================================
# Main
# =============================================================================

def load_data_with_event_matching(path_to_data, events_file, path_to_event_markers = "", exact_event_matching = True, newLabels = True): 

    '''
     """The function reads the data from a specified path and formats it into
     the required structure for further processing.
    Parameters
    ----------
    path_to_data : str
        Path to the .csv file that contains the recorded data.
    events_file : boolean
        True/False whether event markers data is provided.
    path_to_event_markers : str
        Path to the .json file that contains the event_markers.  
    
    Returns
    -------
    df : pandas.DataFrame
        Formatted dataframe containing the sensors data.
    """
    '''
    _file = open(path_to_data, 'r')
    data = _file.read()
    _file.close()
    
    metadata = [line for line in data.split('\n') if '#' in line]

    for line in metadata:
        if line.find('Frame#') == -1:
            data=data.replace("{}".format(line),'', 1)
        if line.find('#Time/Seconds.referenceOffset') != -1:
            time_offset = float(line.split(',')[1])
        if line.find('#Emg/Properties.rawToVoltageDivisor') != -1:
            emg_divisor = float(line.split(',')[1])
        if line.find('#Emg/Properties.contactToImpedanceDivisor') != -1:
            impedance_divisor = float(line.split(',')[1])
        if line.find('#Imu/Properties.accelerationDivisor') != -1:
            acceleration_divisor = float(line.split(',')[1])
        if line.find('#Imu/Properties.magnetometerDivisor') != -1:
            magnetometer_divisor = float(line.split(',')[1])
        if line.find('#Imu/Properties.gyroscopeDivisor') != -1:
            gyroscope_divisor = float(line.split(',')[1])
        #New labels for IMU    
        if line.find('#Accelerometer/Properties.rawDivisor') != -1:
            acceleration_divisor = float(line.split(',')[1])
        if line.find('#Magnetometer/Properties.rawDivisor') != -1:
            magnetometer_divisor = float(line.split(',')[1])
        if line.find('#Gyroscope/Properties.rawDivisor') != -1:
            gyroscope_divisor = float(line.split(',')[1])
    data = re.sub(r'\n\s*\n', '\n', data, re.MULTILINE)
    data = re.sub(r'\s*\n\s*Frame#','Frame#', data, re.MULTILINE)
    

    df = pd.read_csv(StringIO(data), skip_blank_lines=True, delimiter = ',', na_filter=False)
    df = df.dropna()
    
    #convert raw values to appropriate unit
    if (newLabels):  
        df[['Emg/Amplitude[RightFrontalis]', 'Emg/Amplitude[RightZygomaticus]','Emg/Amplitude[RightOrbicularis]','Emg/Amplitude[CenterCorrugator]','Emg/Amplitude[LeftOrbicularis]','Emg/Amplitude[LeftZygomaticus]','Emg/Amplitude[LeftFrontalis]']] = df[['Emg/Amplitude[RightFrontalis]', 'Emg/Amplitude[RightZygomaticus]','Emg/Amplitude[RightOrbicularis]','Emg/Amplitude[CenterCorrugator]','Emg/Amplitude[LeftOrbicularis]','Emg/Amplitude[LeftZygomaticus]','Emg/Amplitude[LeftFrontalis]']].astype('float') / emg_divisor
        df[['Emg/Filtered[RightFrontalis]', 'Emg/Filtered[RightZygomaticus]', 'Emg/Filtered[RightOrbicularis]','Emg/Filtered[CenterCorrugator]','Emg/Filtered[LeftOrbicularis]','Emg/Filtered[LeftZygomaticus]','Emg/Filtered[LeftFrontalis]']] = df[['Emg/Filtered[RightFrontalis]', 'Emg/Filtered[RightZygomaticus]', 'Emg/Filtered[RightOrbicularis]','Emg/Filtered[CenterCorrugator]','Emg/Filtered[LeftOrbicularis]','Emg/Filtered[LeftZygomaticus]','Emg/Filtered[LeftFrontalis]']].astype('float') / emg_divisor
        df[['Emg/Raw[RightFrontalis]','Emg/Raw[RightZygomaticus]','Emg/Raw[RightOrbicularis]','Emg/Raw[CenterCorrugator]','Emg/Raw[LeftOrbicularis]','Emg/Raw[LeftZygomaticus]','Emg/Raw[LeftFrontalis]']] = df[['Emg/Raw[RightFrontalis]','Emg/Raw[RightZygomaticus]','Emg/Raw[RightOrbicularis]','Emg/Raw[CenterCorrugator]','Emg/Raw[LeftOrbicularis]','Emg/Raw[LeftZygomaticus]','Emg/Raw[LeftFrontalis]']].astype('float') / emg_divisor
        df[['Emg/Contact[RightFrontalis]', 'Emg/Contact[RightZygomaticus]','Emg/Contact[RightOrbicularis]','Emg/Contact[CenterCorrugator]','Emg/Contact[LeftOrbicularis]','Emg/Contact[LeftZygomaticus]','Emg/Contact[LeftFrontalis]']] = df[['Emg/Contact[RightFrontalis]', 'Emg/Contact[RightZygomaticus]','Emg/Contact[RightOrbicularis]','Emg/Contact[CenterCorrugator]','Emg/Contact[LeftOrbicularis]','Emg/Contact[LeftZygomaticus]','Emg/Contact[LeftFrontalis]']].astype('float') / impedance_divisor
        df[['Accelerometer/Raw.x','Accelerometer/Raw.y','Accelerometer/Raw.z']] = df[['Accelerometer/Raw.x','Accelerometer/Raw.y','Accelerometer/Raw.z']].astype('float') / acceleration_divisor / 9.699466695345983
        df[['Magnetometer/Raw.x', 'Magnetometer/Raw.y','Magnetometer/Raw.z']] = df[['Magnetometer/Raw.x', 'Magnetometer/Raw.y','Magnetometer/Raw.z']].astype('float') / magnetometer_divisor
        df[['Gyroscope/Raw.x', 'Gyroscope/Raw.y', 'Gyroscope/Raw.z']] = df[['Gyroscope/Raw.x', 'Gyroscope/Raw.y', 'Gyroscope/Raw.z']].astype('float') / gyroscope_divisor  
    else:
        df[['Emg/Amplitude[0]', 'Emg/Amplitude[1]','Emg/Amplitude[2]','Emg/Amplitude[3]','Emg/Amplitude[4]','Emg/Amplitude[5]','Emg/Amplitude[6]']] = df[['Emg/Amplitude[0]', 'Emg/Amplitude[1]','Emg/Amplitude[2]','Emg/Amplitude[3]','Emg/Amplitude[4]','Emg/Amplitude[5]','Emg/Amplitude[6]']].astype('float') / emg_divisor
        df[['Emg/Filtered[0]', 'Emg/Filtered[1]', 'Emg/Filtered[2]','Emg/Filtered[3]','Emg/Filtered[4]','Emg/Filtered[5]','Emg/Filtered[6]']] = df[['Emg/Filtered[0]', 'Emg/Filtered[1]', 'Emg/Filtered[2]','Emg/Filtered[3]','Emg/Filtered[4]','Emg/Filtered[5]','Emg/Filtered[6]']].astype('float') / emg_divisor
        df[['Emg/Raw[0]','Emg/Raw[1]','Emg/Raw[2]','Emg/Raw[3]','Emg/Raw[4]','Emg/Raw[5]','Emg/Raw[6]']] = df[['Emg/Raw[0]','Emg/Raw[1]','Emg/Raw[2]','Emg/Raw[3]','Emg/Raw[4]','Emg/Raw[5]','Emg/Raw[6]']].astype('float') / emg_divisor
        df[['Emg/Contact[0]', 'Emg/Contact[1]','Emg/Contact[2]','Emg/Contact[3]','Emg/Contact[4]','Emg/Contact[5]','Emg/Contact[6]']] = df[['Emg/Contact[0]', 'Emg/Contact[1]','Emg/Contact[2]','Emg/Contact[3]','Emg/Contact[4]','Emg/Contact[5]','Emg/Contact[6]']].astype('float') / impedance_divisor
        df[['Imu/Accelerometer.x','Imu/Accelerometer.y','Imu/Accelerometer.z']] = df[['Imu/Accelerometer.x','Imu/Accelerometer.y','Imu/Accelerometer.z']].astype('float') / acceleration_divisor / 9.699466695345983
        df[['Imu/Magnetometer.x', 'Imu/Magnetometer.y','Imu/Magnetometer.z']] = df[['Imu/Magnetometer.x', 'Imu/Magnetometer.y','Imu/Magnetometer.z']].astype('float') / magnetometer_divisor
        df[['Imu/Gyroscope.x', 'Imu/Gyroscope.y', 'Imu/Gyroscope.z']] = df[['Imu/Gyroscope.x', 'Imu/Gyroscope.y', 'Imu/Gyroscope.z']].astype('float') / gyroscope_divisor  
    df[['HeartRate/Average']] = df[['HeartRate/Average']].astype('float') 

    unix_timestamp = time_offset + 946684800
    df_timestamps = df['Time'].astype('float')
    


            
    # i=0
    # for index, value in enumerate(df_timestamps):
    #     i = i+1
    #     if(value>2**31): #due to DAB clock running fast, time is negative for the first few rows causing int32 underflow
    #         df_timestamps[index] = value - 2**32
    
        
    #print('Number of iterations: ' + str(i))
    #df['Time'] = df_timestamps
    
    #append event markers (from json) to sensor data dataframe
    if (events_file):

            
            df['unix_timestamp'] = df_timestamps + unix_timestamp 
            
            with open(path_to_event_markers) as f:    
                data=json.load(f)
                df_events = pd.DataFrame()
                events = []
                timestamps = []
        
                for i in range(0, len(data)):
                    events.append(data[i]['Event'])
                    timestamps.append(data[i]['Timestamp'])
                    
                    
            #If events have pairs
            if (not exact_event_matching):        
                 
                #events should be paired: event_1_start - event_1_end; otherwise this will fail
                timestamps = np.array(timestamps).reshape(int(len(timestamps)/2), 2)
                events = events[0::2]
            
                df_events['event'] = events
                df_events = pd.concat([df_events, pd.DataFrame(timestamps, columns=['start', 'end'])], axis=1)
                df_events['start'] = df_events['start']/1000 + 946684800
                df_events['end'] = df_events['end']/1000 + 946684800
                df_events['event'] = df_events['event'].str.split(" ").str.get(0)
                
                for idx, event in enumerate(df_events.iloc[:,0]):
                    df.loc[(df.unix_timestamp >= df_events.iloc[idx, 1]) & 
                             (df.unix_timestamp <= df_events.iloc[idx, 2]), 'event'] = event
                   
            #Else find nearest data timestamp to event and append event to that data row
            else:
                df['Event'] = ''
                for i in range(0, len(events)):
                    timestamp = timestamps[i]/1000 + 946684800
                    event_row_index = df['unix_timestamp'].searchsorted(timestamp) #search for the closest matching timestamp between data and event
                    df['Event'][event_row_index] = events[i]

                    
           # df.drop('unix_timestamp',axis=1,inplace=True)
           
               
    
    return df


def __get_metadata_value(df, key):
    """
    Get metadata from array
    """
    try:
        val = df[ df["metadata"]==key ]["value"]
        return None if (val.size == 0) else val
    except:
        return None

def apply_normalization_from_metadata(data:pd.DataFrame, metadata:pd.DataFrame, verbose:bool = False):
    """
    Applies the normalization values to each of the physiological
    variables collected from the emteqPro mask.

    `data` and `metadata` are generated from the function `load_single_csv_data()`

    Normalization happens based on guidelines from :
        > https://support.emteqlabs.com/data/CSV.html
    """

    # Which variable to look in the metadata, and to which data columns it will be applied.
    NORMALIZATION_TABLE = {
            ### '#Time/Seconds.referenceOffset' = [], # This is always normalized manually to Unix in `load_single_csv_data()`
            '#Emg/Properties.rawToVoltageDivisor': config.EMG_SIGNAL_COLNAMES,
            '#Emg/Properties.contactToImpedanceDivisor': config.EMG_CONTACT_COLNAMES,
            '#Accelerometer/Properties.rawDivisor': config.ACC_COLNAMES,
            '#Magnetometer/Properties.rawDivisor': config.MAG_COLNAMES,
            '#Gyroscope/Properties.rawDivisor': config.GYR_COLNAMES
    }

    for norm_id, cols_to_normalize in NORMALIZATION_TABLE.items():
        norm_value = float( __get_metadata_value(metadata, norm_id) )
        if (verbose): print(f"\t> Normalizing ({norm_id}={norm_value}) in columns: {cols_to_normalize}: ")
        for col in cols_to_normalize:
            # Sometimes the dataframe does not include all variables
            # from the normalization table.
            if(col in data.columns):
                data[col] = data[col] / norm_value
    return data


def load_single_csv_data(path_to_csv:str, 
                                columns:list=None, 
                                filter_wrong_timestamps:bool=True,
                                normalize_reference_time:bool = True,
                                filter_duplicates:bool = True,
                                normalize_from_metadata:bool=True):
    """
    Filepath to CSV file to load.

    :param path_to_csv: Full path to CSV file to be loaded
    :param columns: Subset of columns to extract. You may use `EmgPathMuscles` to generate the list
    :param filter_wrong_timestamps: Remove the rows that contain timestamps <0 and >last_timestamp_in_file
    :param normalize_reference_time: Convert the timestamps in file to Unix using metadata "#Time/Seconds.unixOffset"
    :param filter_duplicates: Removes the duplicate rows from the pandas DataFrame, exclusing the timestamps in the index.

    :return: Data and metadata
    :rtype: A tuple with two pandas.DataFrames
    """

    # Metadata with the character '#'
    metadata = None
    try:
        metadata = pd.read_csv( path_to_csv, sep=",", engine="c", on_bad_lines='skip', header=None, names = ["metadata","value","more"])
        metadata = metadata[ metadata["metadata"].str.startswith("#") ] # Keep rows that start with '#'
    except:
        # Hack needed because some files (e.g., `participant_268/video_2.csv`) have CSV data different than the rest and metadata was not loaded.
        metadata = pd.read_csv( path_to_csv, sep=",", engine="c", on_bad_lines='skip', header=None)#, names = ["metadata","value","more"])
        metadata = metadata[ metadata[ metadata.columns[0] ].str.startswith("#").replace(np.nan, False) ] # Keep rows that start with '#'
        metadata.insert(0, "metadata", metadata[metadata.columns[0]])
        metadata.insert(1, "value", metadata[metadata.columns[2]])
    
    ## Code below does not work. Indexing through strings raises an error in pandas
    # metadata["metadata"]=metadata["metadata"].apply(lambda x: x[1:]) # Remove the symbol # from the beginning of the line
    # metadata.set_index("metadata", inplace=True) # Errors having string as index (To be checked!)
    
    # All lines that do not start with the character '#', therefore `comment="#"`
    data = pd.read_csv( path_to_csv, sep=",", comment="#", engine="c", header=0, names=config.DATA_HEADER_CSV)

    # Convert values to corresponding units based on metadata
    if(normalize_from_metadata):
        data = apply_normalization_from_metadata(data, metadata)

    # Subselect some columns
    if columns is not None:
        columns = list(set(columns)) # Avoid duplicate colnames
        data = data[ [config.TIME_COLNAME] + columns ]

    # Filter data with invalid timestamps
    if(filter_wrong_timestamps):
        # Some timestamps carry over wrong timestamps due to high-freq data, 
        # thus remove samples with values greater than time in the last row
        data = data[ (data[config.TIME_COLNAME] < data[config.TIME_COLNAME].iloc[-1]) ]

    # Time as index in the DF
    data.set_index(config.TIME_COLNAME, inplace=True)

    # Convert timestamps
    if (normalize_reference_time):
        # # Extract unix offset from metadata and add it to data
        _ref_offset = float(__get_metadata_value(metadata,"#Time/Seconds.unixOffset"))
        data.index += _ref_offset # Transform from secs to msec
        # Convert from J2000 to unix
        # _ref_offset = float(__get_metadata_value(metadata,"#Time/Seconds.referenceOffset"))
        # data.index += _ref_offset # Transform from secs to msec

    # Most data contains duplicates because the raw data is not always @2KHz
    if filter_duplicates:
        data.drop_duplicates(keep="first", inplace=True)

    return data, metadata

