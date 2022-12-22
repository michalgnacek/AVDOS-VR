# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:34:01 2022

@author: Michal
"""
#%%
from pathlib import Path
import sys,os
this_path = None
try:    # WORKS WITH .py
    this_path = str(os.path.dirname(os.path.abspath(__file__)))
except: # WORKS WITH .ipynb
    this_path = str(Path().absolute())+"/" 
print("File Path:", this_path)

# Add the level up to the file path so it recognizes the scripts inside `avdosvr`
sys.path.append(os.path.join(this_path, ".."))

#%%

import os
import numpy as np
import pandas as pd

import avdosvr
from avdosvr.verify_data import verify_data
from avdosvr.plots.plots import plot_sm_ppg, plot_fit_state
from avdosvr.utils.classes import Participant
from avdosvr.utils.files_handler import generate_complete_path

from avdosvr.analysis.average_ratings import calculate_average_arousal_valence_ratings, get_average_av_ratings_columns
from avdosvr.analysis.durations import calculate_good_signal_quality_duration, calculate_signal_quality_check_duration, get_durations_columns
from avdosvr.analysis.skip_participants import skip_participant

import avdosvr.hrv.HRV_analysis_cloud as HRV

# Hide warning about pandas Slicing
pd.options.mode.chained_assignment = None

#%% Set variables

#Directory containing participant data downloaded from gnacek.com/
data_directory = (r"../data")
# Identifier for the notebook when creating temp files
NOTEBOOK_NAME = "0_verify/" 

#Bool variables to enable/disable processing of segments of the data to speed up the runtime
process_slow_movement = False
process_fast_movement = False
process_video = True

validate_data = True

calculate_duration = True
calculate_average_av_ratings = True

#Bool variables to enable/disable plotting of graphs
plots_output_root_folder = "./temp/"+NOTEBOOK_NAME
plot_slow_movement_ppg = False
plot_contact_fit_state = False

no_of_frames_to_drop_from_start_of_recording = 1001

#%%

# Data for each participant is stored within a folder. Names of files indicate which segment of remote study it is
list_data_dir = os.listdir(data_directory)
is_folder = [os.path.isdir(os.path.join(data_directory,d)) for d in list_data_dir]
participants_list = np.array(list_data_dir)[ is_folder ]

#%% Get list of participant folders/files
print("Found data folders for " + str(len(participants_list)) + " participants")

#%% Run data validation check on all participants
if(validate_data):
    fit_states_signal_quality = None
    for participant in participants_list:
        new_df = verify_data(data_directory + "/" + participant)
        # fit_states_signal_quality = new_df if (fit_states_signal_quality is None) else pd.concat([fit_states_signal_quality,new_df], axis=0)
       #  if (fit_states_signal_quality is None):
          #  fit_states_signal_quality = new_df
      #  else:
         #   pd.concat([fit_states_signal_quality,new_df], axis=0)
        #fit_states_signal_quality = new_df if (fit_states_signal_quality is None) else pd.concat([fit_states_signal_quality,new_df], axis=0)
    print("----FINISHED DATA CHECK FOR ALL PARTICIPANTS SEE ABOVE FOR OUTPUT----")

#%% Run data validation check on all participants
if(validate_data):
    fit_states_signal_quality = None
    for participant in participants_list:
        new_df = verify_data(data_directory + "/" + participant)
        # fit_states_signal_quality = new_df if (fit_states_signal_quality is None) else pd.concat([fit_states_signal_quality,new_df], axis=0)
        if (fit_states_signal_quality is None):
            fit_states_signal_quality = new_df.copy()
        else:
            fit_states_signal_quality = pd.concat([fit_states_signal_quality,new_df], axis=0, ignore_index=True)
        #fit_states_signal_quality = new_df if (fit_states_signal_quality is None) else pd.concat([fit_states_signal_quality,new_df], axis=0)
    print("----FINISHED DATA CHECK FOR ALL PARTICIPANTS SEE ABOVE FOR OUTPUT----")
    
#%%
def drop_start_frames(data, frames_to_drop):
    return data.drop(data.index[0:frames_to_drop])

#%% Main analysis script

if(calculate_duration):
    durations = pd.DataFrame(columns = get_durations_columns())
if(calculate_average_av_ratings):
     average_av_ratings = pd.DataFrame(columns = get_average_av_ratings_columns())

participant_counter = 1
# debug_counter = 1

#Loop through all participant data
for participant in participants_list:
    # Processing slow movement data
    # if(participant_counter==1):
    #     participant = participants_list[debug_counter]
    # else:
    #     participant = participants_list[debug_counter+participant_counter]
    ParticipantObj = Participant(participant, data_directory)

    print("Processing data for: " + ParticipantObj.name + ". " + str(participant_counter) + " out of " + str(len(participants_list)))
    ParticipantObj = skip_participant(ParticipantObj)
    
    if process_video:
        if(ParticipantObj.skip_video == False):
            print("Video Start")
            if(ParticipantObj.skip_video_1 == False):
                video_1_data = ParticipantObj.getVideo1Data()
                video_1_events = video_1_data[video_1_data['Event'] != '']
                video_1_data = drop_start_frames(video_1_data, no_of_frames_to_drop_from_start_of_recording)
                if(calculate_duration):
                    video_1_duration = round((video_1_data['Time'].iloc[-1] - video_1_data['Time'].iloc[0]),3)
                    video_1_GoodDuration, video_1_TimeTakenToEstablishGoodSignal = calculate_good_signal_quality_duration(video_1_data)
                    video_1_SignalCheckDuration = calculate_signal_quality_check_duration(video_1_events)
                    video_1_DataExcludingSignalCheck = video_1_data.drop(video_1_data.index[0:video_1_events['Frame#'].iloc[1]-1])
                    video_1_DurationExcludingSignalCheck = round((video_1_DataExcludingSignalCheck['Time'].iloc[-1] - video_1_DataExcludingSignalCheck['Time'].iloc[0]),3)
                    video_1_GoodDurationExcludingSignalCheck, blank = calculate_good_signal_quality_duration(video_1_data.drop(video_1_data.index[0:video_1_events['Frame#'].iloc[1]-1])) #drop 
                if plot_contact_fit_state:
                    plot_fit_state(video_1_data, video_1_events, ParticipantObj.name, "Video 1", root_folder=plots_output_root_folder)
            else:
                print("Skipping VIDEO_1 for:" + ParticipantObj.name)
                video_1_duration = "SKIPPED" 
                video_1_GoodDuration = "SKIPPED"  
                video_1_TimeTakenToEstablishGoodSignal = "SKIPPED" 
                video_1_SignalCheckDuration = "SKIPPED" 
                video_1_DurationExcludingSignalCheck = "SKIPPED" 
                video_1_GoodDurationExcludingSignalCheck = "SKIPPED"
                
            if(ParticipantObj.skip_video_2 == False):
                video_2_data = ParticipantObj.getVideo2Data()
                video_2_events = video_2_data[video_2_data['Event'] != '']
                if(calculate_average_av_ratings):
                    video_2_ratings = calculate_average_arousal_valence_ratings(video_2_events)
                video_2_data = drop_start_frames(video_2_data, no_of_frames_to_drop_from_start_of_recording)
                if(calculate_duration):
                    video_2_duration = video_2_data['Time'].iloc[-1] - video_2_data['Time'].iloc[0]
                    video_2_GoodDuration, blank = calculate_good_signal_quality_duration(video_2_data)
                if plot_contact_fit_state:
                    plot_fit_state(video_2_data, video_2_events, ParticipantObj.name, "Video 2", root_folder=plots_output_root_folder)
            else:
                print("Skipping VIDEO_2 for:" + ParticipantObj.name)
                video_2_duration = "SKIPPED" 
                video_2_GoodDuration = "SKIPPED" 
                
            if(ParticipantObj.skip_video_3 == False):
                video_3_data = ParticipantObj.getVideo3Data()
                video_3_events = video_3_data[video_3_data['Event'] != '']
                if(calculate_average_av_ratings):
                    video_3_ratings = calculate_average_arousal_valence_ratings(video_3_events)
                video_3_data = drop_start_frames(video_3_data, no_of_frames_to_drop_from_start_of_recording)
                if(calculate_duration):
                    video_3_duration = video_3_data['Time'].iloc[-1] - video_3_data['Time'].iloc[0]
                    video_3_GoodDuration, blank = calculate_good_signal_quality_duration(video_3_data)
                if plot_contact_fit_state:
                    plot_fit_state(video_3_data, video_3_events, ParticipantObj.name, "Video 3", root_folder=plots_output_root_folder)
            else:
                print("Skipping VIDEO_3 for:" + ParticipantObj.name)
                video_3_duration = "SKIPPED" 
                video_3_GoodDuration = "SKIPPED" 
                
            if(ParticipantObj.skip_video_4 == False):
                video_4_data = ParticipantObj.getVideo4Data()
                video_4_events = video_4_data[video_4_data['Event'] != '']
                if(calculate_average_av_ratings):
                    video_4_ratings = calculate_average_arousal_valence_ratings(video_4_events)
                video_4_data = drop_start_frames(video_4_data, no_of_frames_to_drop_from_start_of_recording)
                if(calculate_duration):
                    video_4_duration = video_4_data['Time'].iloc[-1] - video_4_data['Time'].iloc[0]
                    video_4_GoodDuration, blank = calculate_good_signal_quality_duration(video_4_data)
                if plot_contact_fit_state:
                    plot_fit_state(video_4_data, video_4_events, ParticipantObj.name, "Video 4", root_folder=plots_output_root_folder)
            else:
                print("Skipping VIDEO_4 for:" + ParticipantObj.name)
                video_4_duration = "SKIPPED" 
                video_4_GoodDuration = "SKIPPED" 

            if(ParticipantObj.skip_video_5 == False):
                video_5_data = ParticipantObj.getVideo5Data()
                video_5_events = video_5_data[video_5_data['Event'] != '']
                if(calculate_average_av_ratings):
                    video_5_ratings = calculate_average_arousal_valence_ratings(video_5_events)
                video_5_data = drop_start_frames(video_5_data, no_of_frames_to_drop_from_start_of_recording)
                if(calculate_duration):
                    video_5_duration = video_5_data['Time'].iloc[-1] - video_5_data['Time'].iloc[0]
                    video_5_GoodDuration, blank = calculate_good_signal_quality_duration(video_5_data)
                if plot_contact_fit_state:
                    plot_fit_state(video_5_data, video_5_events, ParticipantObj.name, "Video 5", root_folder=plots_output_root_folder)
            else:
                print("Skipping VIDEO_5 for:" + ParticipantObj.name)
                video_5_duration = "SKIPPED" 
                video_5_GoodDuration = "SKIPPED" 
            
            if(calculate_average_av_ratings):
                current_participant_average_av_ratings = {"participant": ParticipantObj.name, 
                                      "relax_2_valence": None, "relax_2_arousal": None, "relax_2_no_ratings": None, "relax_3_valence": None, "relax_3_arousal": None, "relax_3_no_ratings": None, "relax_4_valence": None, "relax_4_arousal": None, "relax_4_no_ratings": None, "relax_5_valence": None, "relax_5_arousal": None, "relax_5_no_ratings": None,
                                      "video_03_valence": None, "video_03_arousal": None, "video_03_no_ratings": None, "video_04_valence": None, "video_04_arousal": None, "video_04_no_ratings": None, "video_05_valence": None, "video_05_arousal": None, "video_05_no_ratings": None, "video_06_valence": None, "video_06_arousal": None, "video_06_no_ratings": None, "video_10_valence": None, "video_10_arousal": None, "video_10_no_ratings": None, "video_12_valence": None, "video_12_arousal": None, "video_12_no_ratings": None, "video_13_valence": None, "video_13_arousal": None, "video_13_no_ratings": None, "video_18_valence": None, "video_18_arousal": None, "video_18_no_ratings": None, "video_19_valence": None, "video_19_arousal": None, "video_19_no_ratings": None, "video_20_valence": None, "video_20_arousal": None, "video_20_no_ratings": None,
                                      "video_21_valence": None, "video_21_arousal": None, "video_21_no_ratings": None, "video_22_valence": None, "video_22_arousal": None, "video_22_no_ratings": None, "video_23_valence": None, "video_23_arousal": None, "video_23_no_ratings": None, "video_25_valence": None, "video_25_arousal": None, "video_25_no_ratings": None, "video_29_valence": None, "video_29_arousal": None, "video_29_no_ratings": None, "video_31_valence": None, "video_31_arousal": None, "video_31_no_ratings": None, "video_33_valence": None, "video_33_arousal": None, "video_33_no_ratings": None, "video_37_valence": None, "video_37_arousal": None, "video_37_no_ratings": None, "video_38_valence": None, "video_38_arousal": None, "video_38_no_ratings": None, "video_39_valence": None, "video_39_arousal": None, "video_39_no_ratings": None,
                                      "video_41_valence": None, "video_41_arousal": None, "video_41_no_ratings": None, "video_42_valence": None, "video_42_arousal": None, "video_42_no_ratings": None, "video_46_valence": None, "video_46_arousal": None, "video_46_no_ratings": None, "video_48_valence": None, "video_48_arousal": None, "video_48_no_ratings": None, "video_49_valence": None, "video_49_arousal": None, "video_49_no_ratings": None, "video_51_valence": None, "video_51_arousal": None, "video_51_no_ratings": None, "video_55_valence": None, "video_55_arousal": None, "video_55_no_ratings": None, "video_56_valence": None, "video_56_arousal": None, "video_56_no_ratings": None, "video_57_valence": None, "video_57_arousal": None, "video_57_no_ratings": None, "video_58_valence": None, "video_58_arousal": None, "video_58_no_ratings": None}  
                current_participant_average_av_ratings['relax_2_valence'] = video_2_ratings[0].average_valence
                current_participant_average_av_ratings['relax_2_arousal'] = video_2_ratings[0].average_arousal
                current_participant_average_av_ratings['relax_2_no_ratings'] = len(video_2_ratings[0].arousal_ratings)
                current_participant_average_av_ratings['relax_3_valence'] = video_3_ratings[0].average_valence
                current_participant_average_av_ratings['relax_3_arousal'] = video_3_ratings[0].average_arousal
                current_participant_average_av_ratings['relax_3_no_ratings'] = len(video_3_ratings[0].arousal_ratings)
                current_participant_average_av_ratings['relax_4_valence'] = video_4_ratings[0].average_valence
                current_participant_average_av_ratings['relax_4_arousal'] = video_4_ratings[0].average_arousal
                current_participant_average_av_ratings['relax_4_no_ratings'] = len(video_4_ratings[0].arousal_ratings)
                current_participant_average_av_ratings['relax_5_valence'] = video_5_ratings[0].average_valence
                current_participant_average_av_ratings['relax_5_arousal'] = video_5_ratings[0].average_arousal
                current_participant_average_av_ratings['relax_5_no_ratings'] = len(video_5_ratings[0].arousal_ratings)
                
                combined_affective_video_ratings = video_2_ratings + video_3_ratings + video_4_ratings
                for video_rating in combined_affective_video_ratings:
                    if(video_rating.name!="relax"):
                        current_participant_average_av_ratings["video_" + video_rating.name + "_valence"] = video_rating.average_valence
                        current_participant_average_av_ratings["video_" + video_rating.name + "_arousal"] = video_rating.average_arousal
                        current_participant_average_av_ratings["video_" + video_rating.name + "_no_ratings"] = len(video_rating.arousal_ratings)

                average_av_ratings = average_av_ratings.append(current_participant_average_av_ratings, ignore_index = True)
                # average_av_ratings = pd.concat([average_av_ratings, current_participant_average_av_ratings], axis=0, ignore_index = True)
                
            print("Video End")
        else:
            print("Skipping VIDEO for:" + ParticipantObj.name)
            video_1_duration = "SKIPPED" 
            video_1_GoodDuration = "SKIPPED"  
            video_1_TimeTakenToEstablishGoodSignal = "SKIPPED" 
            video_1_SignalCheckDuration = "SKIPPED" 
            video_1_DurationExcludingSignalCheck = "SKIPPED" 
            video_1_GoodDurationExcludingSignalCheck = "SKIPPED" 
            video_2_duration = "SKIPPED" 
            video_2_GoodDuration = "SKIPPED" 
            video_3_duration = "SKIPPED" 
            video_3_GoodDuration = "SKIPPED" 
            video_4_duration = "SKIPPED" 
            video_4_GoodDuration = "SKIPPED" 
            video_5_duration = "SKIPPED" 
            video_5_GoodDuration = "SKIPPED" 
    
    if(calculate_duration):
        current_participant_durations = {'participant': ParticipantObj.name,
                                        #  'slow_movement_total_duration': slowMovementDuration,'slow_movement_total_good_fit': slowMovementGoodDuration, 'slowMovementTimeTakenToEstablishGoodSignal': slowMovementTimeTakenToEstablishGoodSignal,'slow_movement_signal_check_duration': slowMovementSignalCheckDuration,'slow_movement_duration_excluding_signal_check': slowMovementDurationExcludingSignalCheck,'slow_movement_good_fit_excluding_signal_check': slowMovementGoodDurationExcludingSignalCheck,
                                        #  'fast_movement_total_duration': fastMovementDuration,'fast_movement_total_good_fit': fastMovementGoodDuration, 'fastMovementTimeTakenToEstablishGoodSignal': fastMovementTimeTakenToEstablishGoodSignal,'fast_movement_signal_check_duration': fastMovementSignalCheckDuration,'fast_movement_duration_excluding_signal_check': fastMovementDurationExcludingSignalCheck,'fast_movement_good_fit_excluding_signal_check': fastMovementGoodDurationExcludingSignalCheck,
                                         'video_1_total_duration': video_1_duration,'video_1_total_good_fit': video_1_GoodDuration, 'video_1_TimeTakenToEstablishGoodSignal': video_1_TimeTakenToEstablishGoodSignal,'video_1_signal_check_duration': video_1_SignalCheckDuration,'video_1_duration_excluding_signal_check': video_1_DurationExcludingSignalCheck,'video_1_good_fit_excluding_signal_check': video_1_GoodDurationExcludingSignalCheck,
                                         'video_2_duration': video_2_duration, 'video_2_good_fit': video_2_GoodDuration,
                                         'video_3_duration': video_3_duration, 'video_3_good_fit': video_3_GoodDuration,
                                         'video_4_duration': video_4_duration, 'video_4_good_fit': video_4_GoodDuration,
                                         'video_5_duration': video_5_duration, 'video_5_good_fit': video_5_GoodDuration}

        durations = durations.append(current_participant_durations, ignore_index = True)
        # durations = pd.concat([durations,current_participant_durations], axis=0, ignore_index = True)
   
    participant_counter = participant_counter + 1
print("Finished processing data for: " + ParticipantObj.name)


#%% Main analysis script

if(calculate_duration):
    durations = pd.DataFrame(columns = get_durations_columns())
if(calculate_average_av_ratings):
     average_av_ratings = pd.DataFrame(columns = get_average_av_ratings_columns())

participant_counter = 1

#Set it to participant index (0-38) if you wish to start at specific participant
custom_start_index = -1

#Loop through all participant data
for participant in participants_list:
    
    #Run for specific participant
    if(custom_start_index!=-1):
        if(participant_counter==1):
            participant = participants_list[custom_start_index]
        else:
            participant = participants_list[custom_start_index+participant_counter]
            
    ParticipantObj = Participant(participant, data_directory)

    print("Processing data for: " + ParticipantObj.name + ". " + str(participant_counter) + " out of " + str(len(participants_list)))
    ParticipantObj = skip_participant(ParticipantObj)
    
    if process_video:
        if(ParticipantObj.skip_video == False):
            print("Video Start")
            if(ParticipantObj.skip_video_1 == False):
                video_1_data = ParticipantObj.getVideo1Data()
                video_1_events = video_1_data[video_1_data['Event'] != '']
                video_1_data = drop_start_frames(video_1_data, no_of_frames_to_drop_from_start_of_recording)
                if(calculate_duration):
                    video_1_duration = round((video_1_data['Time'].iloc[-1] - video_1_data['Time'].iloc[0]),3)
                    video_1_GoodDuration, video_1_TimeTakenToEstablishGoodSignal = calculate_good_signal_quality_duration(video_1_data)
                    video_1_SignalCheckDuration = calculate_signal_quality_check_duration(video_1_events)
                    video_1_DataExcludingSignalCheck = video_1_data.drop(video_1_data.index[0:video_1_events['Frame#'].iloc[1]-1])
                    video_1_DurationExcludingSignalCheck = round((video_1_DataExcludingSignalCheck['Time'].iloc[-1] - video_1_DataExcludingSignalCheck['Time'].iloc[0]),3)
                    video_1_GoodDurationExcludingSignalCheck, blank = calculate_good_signal_quality_duration(video_1_data.drop(video_1_data.index[0:video_1_events['Frame#'].iloc[1]-1])) #drop 
                if plot_contact_fit_state:
                    plot_fit_state(video_1_data, video_1_events, ParticipantObj.name, "Video 1", root_folder=plots_output_root_folder)
            else:
                print("Skipping VIDEO_1 for:" + ParticipantObj.name)
                video_1_duration = "SKIPPED" 
                video_1_GoodDuration = "SKIPPED"  
                video_1_TimeTakenToEstablishGoodSignal = "SKIPPED" 
                video_1_SignalCheckDuration = "SKIPPED" 
                video_1_DurationExcludingSignalCheck = "SKIPPED" 
                video_1_GoodDurationExcludingSignalCheck = "SKIPPED"
                
            if(ParticipantObj.skip_video_2 == False):
                video_2_data = ParticipantObj.getVideo2Data()
                video_2_events = video_2_data[video_2_data['Event'] != '']
                if(calculate_average_av_ratings):
                    video_2_ratings = calculate_average_arousal_valence_ratings(video_2_events)
                video_2_data = drop_start_frames(video_2_data, no_of_frames_to_drop_from_start_of_recording)
                if(calculate_duration):
                    video_2_duration = video_2_data['Time'].iloc[-1] - video_2_data['Time'].iloc[0]
                    video_2_GoodDuration, blank = calculate_good_signal_quality_duration(video_2_data)
                if plot_contact_fit_state:
                    plot_fit_state(video_2_data, video_2_events, ParticipantObj.name, "Video 2", root_folder=plots_output_root_folder)
            else:
                print("Skipping VIDEO_2 for:" + ParticipantObj.name)
                video_2_duration = "SKIPPED" 
                video_2_GoodDuration = "SKIPPED" 
                
            if(ParticipantObj.skip_video_3 == False):
                video_3_data = ParticipantObj.getVideo3Data()
                video_3_events = video_3_data[video_3_data['Event'] != '']
                if(calculate_average_av_ratings):
                    video_3_ratings = calculate_average_arousal_valence_ratings(video_3_events)
                video_3_data = drop_start_frames(video_3_data, no_of_frames_to_drop_from_start_of_recording)
                if(calculate_duration):
                    video_3_duration = video_3_data['Time'].iloc[-1] - video_3_data['Time'].iloc[0]
                    video_3_GoodDuration, blank = calculate_good_signal_quality_duration(video_3_data)
                if plot_contact_fit_state:
                    plot_fit_state(video_3_data, video_3_events, ParticipantObj.name, "Video 3", root_folder=plots_output_root_folder)
            else:
                print("Skipping VIDEO_3 for:" + ParticipantObj.name)
                video_3_duration = "SKIPPED" 
                video_3_GoodDuration = "SKIPPED" 
                
            if(ParticipantObj.skip_video_4 == False):
                video_4_data = ParticipantObj.getVideo4Data()
                video_4_events = video_4_data[video_4_data['Event'] != '']
                if(calculate_average_av_ratings):
                    video_4_ratings = calculate_average_arousal_valence_ratings(video_4_events)
                video_4_data = drop_start_frames(video_4_data, no_of_frames_to_drop_from_start_of_recording)
                if(calculate_duration):
                    video_4_duration = video_4_data['Time'].iloc[-1] - video_4_data['Time'].iloc[0]
                    video_4_GoodDuration, blank = calculate_good_signal_quality_duration(video_4_data)
                if plot_contact_fit_state:
                    plot_fit_state(video_4_data, video_4_events, ParticipantObj.name, "Video 4", root_folder=plots_output_root_folder)
            else:
                print("Skipping VIDEO_4 for:" + ParticipantObj.name)
                video_4_duration = "SKIPPED" 
                video_4_GoodDuration = "SKIPPED" 

            if(ParticipantObj.skip_video_5 == False):
                video_5_data = ParticipantObj.getVideo5Data()
                video_5_events = video_5_data[video_5_data['Event'] != '']
                if(calculate_average_av_ratings):
                    video_5_ratings = calculate_average_arousal_valence_ratings(video_5_events)
                video_5_data = drop_start_frames(video_5_data, no_of_frames_to_drop_from_start_of_recording)
                if(calculate_duration):
                    video_5_duration = video_5_data['Time'].iloc[-1] - video_5_data['Time'].iloc[0]
                    video_5_GoodDuration, blank = calculate_good_signal_quality_duration(video_5_data)
                if plot_contact_fit_state:
                    plot_fit_state(video_5_data, video_5_events, ParticipantObj.name, "Video 5", root_folder=plots_output_root_folder)
            else:
                print("Skipping VIDEO_5 for:" + ParticipantObj.name)
                video_5_duration = "SKIPPED" 
                video_5_GoodDuration = "SKIPPED" 
            
            if(calculate_average_av_ratings):
                current_participant_average_av_ratings = {"participant": ParticipantObj.name, 
                                      "relax_2_valence": None, "relax_2_arousal": None, "relax_2_no_ratings": None, "relax_3_valence": None, "relax_3_arousal": None, "relax_3_no_ratings": None, "relax_4_valence": None, "relax_4_arousal": None, "relax_4_no_ratings": None, "relax_5_valence": None, "relax_5_arousal": None, "relax_5_no_ratings": None,
                                      "video_03_valence": None, "video_03_arousal": None, "video_03_no_ratings": None, "video_04_valence": None, "video_04_arousal": None, "video_04_no_ratings": None, "video_05_valence": None, "video_05_arousal": None, "video_05_no_ratings": None, "video_06_valence": None, "video_06_arousal": None, "video_06_no_ratings": None, "video_10_valence": None, "video_10_arousal": None, "video_10_no_ratings": None, "video_12_valence": None, "video_12_arousal": None, "video_12_no_ratings": None, "video_13_valence": None, "video_13_arousal": None, "video_13_no_ratings": None, "video_18_valence": None, "video_18_arousal": None, "video_18_no_ratings": None, "video_19_valence": None, "video_19_arousal": None, "video_19_no_ratings": None, "video_20_valence": None, "video_20_arousal": None, "video_20_no_ratings": None,
                                      "video_21_valence": None, "video_21_arousal": None, "video_21_no_ratings": None, "video_22_valence": None, "video_22_arousal": None, "video_22_no_ratings": None, "video_23_valence": None, "video_23_arousal": None, "video_23_no_ratings": None, "video_25_valence": None, "video_25_arousal": None, "video_25_no_ratings": None, "video_29_valence": None, "video_29_arousal": None, "video_29_no_ratings": None, "video_31_valence": None, "video_31_arousal": None, "video_31_no_ratings": None, "video_33_valence": None, "video_33_arousal": None, "video_33_no_ratings": None, "video_37_valence": None, "video_37_arousal": None, "video_37_no_ratings": None, "video_38_valence": None, "video_38_arousal": None, "video_38_no_ratings": None, "video_39_valence": None, "video_39_arousal": None, "video_39_no_ratings": None,
                                      "video_41_valence": None, "video_41_arousal": None, "video_41_no_ratings": None, "video_42_valence": None, "video_42_arousal": None, "video_42_no_ratings": None, "video_46_valence": None, "video_46_arousal": None, "video_46_no_ratings": None, "video_48_valence": None, "video_48_arousal": None, "video_48_no_ratings": None, "video_49_valence": None, "video_49_arousal": None, "video_49_no_ratings": None, "video_51_valence": None, "video_51_arousal": None, "video_51_no_ratings": None, "video_55_valence": None, "video_55_arousal": None, "video_55_no_ratings": None, "video_56_valence": None, "video_56_arousal": None, "video_56_no_ratings": None, "video_57_valence": None, "video_57_arousal": None, "video_57_no_ratings": None, "video_58_valence": None, "video_58_arousal": None, "video_58_no_ratings": None}  
                current_participant_average_av_ratings['relax_2_valence'] = video_2_ratings[0].average_valence
                current_participant_average_av_ratings['relax_2_arousal'] = video_2_ratings[0].average_arousal
                current_participant_average_av_ratings['relax_2_no_ratings'] = len(video_2_ratings[0].arousal_ratings)
                current_participant_average_av_ratings['relax_3_valence'] = video_3_ratings[0].average_valence
                current_participant_average_av_ratings['relax_3_arousal'] = video_3_ratings[0].average_arousal
                current_participant_average_av_ratings['relax_3_no_ratings'] = len(video_3_ratings[0].arousal_ratings)
                current_participant_average_av_ratings['relax_4_valence'] = video_4_ratings[0].average_valence
                current_participant_average_av_ratings['relax_4_arousal'] = video_4_ratings[0].average_arousal
                current_participant_average_av_ratings['relax_4_no_ratings'] = len(video_4_ratings[0].arousal_ratings)
                current_participant_average_av_ratings['relax_5_valence'] = video_5_ratings[0].average_valence
                current_participant_average_av_ratings['relax_5_arousal'] = video_5_ratings[0].average_arousal
                current_participant_average_av_ratings['relax_5_no_ratings'] = len(video_5_ratings[0].arousal_ratings)
                
                combined_affective_video_ratings = video_2_ratings + video_3_ratings + video_4_ratings
                for video_rating in combined_affective_video_ratings:
                    if(video_rating.name!="relax"):
                        current_participant_average_av_ratings["video_" + video_rating.name + "_valence"] = video_rating.average_valence
                        current_participant_average_av_ratings["video_" + video_rating.name + "_arousal"] = video_rating.average_arousal
                        current_participant_average_av_ratings["video_" + video_rating.name + "_no_ratings"] = len(video_rating.arousal_ratings)


                #verage_av_ratings = average_av_ratings.append(current_participant_average_av_ratings, ignore_index = True)
                # average_av_ratings = pd.concat([average_av_ratings, current_participant_average_av_ratings], axis=0, ignore_index = True)
                average_av_ratings = pd.concat([average_av_ratings, pd.DataFrame.from_dict([current_participant_average_av_ratings])], ignore_index = True)
            print("Video End")
        else:
            print("Skipping VIDEO for:" + ParticipantObj.name)
            video_1_duration = "SKIPPED" 
            video_1_GoodDuration = "SKIPPED"  
            video_1_TimeTakenToEstablishGoodSignal = "SKIPPED" 
            video_1_SignalCheckDuration = "SKIPPED" 
            video_1_DurationExcludingSignalCheck = "SKIPPED" 
            video_1_GoodDurationExcludingSignalCheck = "SKIPPED" 
            video_2_duration = "SKIPPED" 
            video_2_GoodDuration = "SKIPPED" 
            video_3_duration = "SKIPPED" 
            video_3_GoodDuration = "SKIPPED" 
            video_4_duration = "SKIPPED" 
            video_4_GoodDuration = "SKIPPED" 
            video_5_duration = "SKIPPED" 
            video_5_GoodDuration = "SKIPPED" 
    
    if(calculate_duration):
        current_participant_durations = {'participant': ParticipantObj.name,
                                         'video_1_total_duration': video_1_duration,'video_1_total_good_fit': video_1_GoodDuration, 'video_1_TimeTakenToEstablishGoodSignal': video_1_TimeTakenToEstablishGoodSignal,'video_1_signal_check_duration': video_1_SignalCheckDuration,'video_1_duration_excluding_signal_check': video_1_DurationExcludingSignalCheck,'video_1_good_fit_excluding_signal_check': video_1_GoodDurationExcludingSignalCheck,
                                         'video_2_duration': video_2_duration, 'video_2_good_fit': video_2_GoodDuration,
                                         'video_3_duration': video_3_duration, 'video_3_good_fit': video_3_GoodDuration,
                                         'video_4_duration': video_4_duration, 'video_4_good_fit': video_4_GoodDuration,
                                         'video_5_duration': video_5_duration, 'video_5_good_fit': video_5_GoodDuration}

        #durations = durations.append(current_participant_durations, ignore_index = True)
        durations = pd.concat([durations,pd.DataFrame.from_dict([current_participant_durations])], ignore_index = True)
   
    participant_counter = participant_counter + 1
print("Finished processing data for: " + ParticipantObj.name)
    
