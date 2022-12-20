# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:27:25 2022

@author: Michal Gnacek
"""
#%% Variables
# Fit State treshold for good signal quality duration calculation. Fit State is an abstract continuous measurement of Mask ‘Fit' with higher values representing the ideal state of system performance/quality (https://support.emteqlabs.com/)
fit_state_treshold = 8

#%% calculates time in seconds where contactFitState was equal to or above 8 (average)
def calculate_good_signal_quality_duration(data):
    good_signal_duration = 0
    time_taken_to_establish_good_signal = 0
    current_signal_good_quality = False

    for index, row in data.iterrows():
        if(current_signal_good_quality==False):
            if(row['Faceplate/FitState']>=fit_state_treshold):
                current_signal_good_quality = True
                good_signal_start_time = row['Time']
                if(good_signal_duration==0):
                    time_taken_to_establish_good_signal = row['Time']-data['Time'].iloc[0]
        else:
            if(row['Faceplate/FitState']<fit_state_treshold):
                current_signal_good_quality = False
                good_signal_end_time = row['Time']
                good_signal_duration = good_signal_duration + (good_signal_end_time-good_signal_start_time)
    if(current_signal_good_quality == True):
        good_signal_duration = good_signal_duration + (data['Time'].iloc[-1]-good_signal_start_time)
    return round(good_signal_duration,3), round(time_taken_to_establish_good_signal,3)

#%%
def calculate_signal_quality_check_duration(events):
    if('Start of signal check' in events['Event'].iloc[0]):
        signal_quality_check_start_time = events['Time'].iloc[0]
        if(signal_quality_check_start_time>2**31): #due to DAB clock running fast, time is negative for the first few rows causing int32 underflow
            signal_quality_check_start_time = signal_quality_check_start_time - 2**32 
    else:
        print('ERROR - No Start Signal Check message found to calculate signal check duration')
    if('Signal check finished' in events['Event'].iloc[1]):
        signal_quality_check_end_time = events['Time'].iloc[1]
    else:
        print('ERROR - No Signal Check Finished message found to calculate signal check duration')
    return round((signal_quality_check_end_time-signal_quality_check_start_time),3)

def get_durations_columns():
    durations_columns = ["participant", 
                          "video_1_total_duration" ,"video_1_total_good_fit", "video_1_TimeTakenToEstablishGoodSignal", "video_1_signal_check_duration","video_1_duration_excluding_signal_check", "video_1_good_fit_excluding_signal_check",
                          "video_2_duration", "video_2_good_fit",
                          "video_3_duration", "video_3_good_fit",
                          "video_4_duration", "video_4_good_fit",
                          "video_5_duration", "video_5_good_fit"]
    return durations_columns
