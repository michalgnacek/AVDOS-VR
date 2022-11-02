# -*- coding: utf-8 -*-
"""
@author: user
"""

import pandas as pd
import numpy as np
import json
import re
from io import StringIO

def load_data(path_to_data, events_file, path_to_event_markers = "", exact_event_matching = True, newLabels = True): 
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