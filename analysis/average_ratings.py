# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:23:02 2022

@author: Michal Gnacek
"""

from classes.Video import Video

#%%
def calculate_average_arousal_valence_ratings(video_events):
    rest_video_start_index = None
    rest_video_end_index = None
    video_counter = 0
    video_start_index = None
    video_end_index = None
    current_video_name = ""
    videos = []
    
    for i in range(0, len(video_events)):
        if(video_events['Event'].iloc[i] == "Playing rest video"):
            rest_video_start_index = i
            current_video_name = "relax"
            for j in range(i, len(video_events)):
                if(video_events['Event'].iloc[j] == "Finished playing rest video"):
                    rest_video_end_index = j
                    video_object = Video(video_events['Event'].iloc[rest_video_start_index:rest_video_end_index+1], current_video_name)
                    videos.append(video_object)
                    video_counter = video_counter + 1
                    current_video_name = ""
                    break
        elif("Playing video number: " in video_events['Event'].iloc[i]):
            video_start_index = i
            current_video_name = video_events['Event'].iloc[i].split("number: ",1)[1]
            for j in range(i, len(video_events)):
                if(("Finished playing video number: "+ current_video_name) in video_events['Event'].iloc[j]):
                    video_end_index =  j
                    video_object = Video(video_events['Event'].iloc[video_start_index:video_end_index+1], current_video_name)
                    videos.append(video_object)
                    video_counter = video_counter + 1
                    video_start_index = video_end_index = None
                    current_video_name = ""
                    break
                
    return videos

#%%
def get_average_av_ratings_columns():
    average_av_ratings_columns = ["participant", 
                                  "relax_2_valence", "relax_2_arousal", "relax_2_no_ratings", "relax_3_valence", "relax_3_arousal", "relax_3_no_ratings", "relax_4_valence", "relax_4_arousal", "relax_4_no_ratings", "relax_5_valence", "relax_5_arousal", "relax_5_no_ratings",
                                  "video_03_valence", "video_03_arousal", "video_03_no_ratings", "video_04_valence", "video_04_arousal", "video_04_no_ratings", "video_05_valence", "video_05_arousal", "video_05_no_ratings", "video_06_valence", "video_06_arousal", "video_06_no_ratings", "video_10_valence", "video_10_arousal", "video_10_no_ratings", "video_12_valence", "video_12_arousal", "video_12_no_ratings", "video_13_valence", "video_13_arousal", "video_13_no_ratings", "video_18_valence", "video_18_arousal", "video_18_no_ratings", "video_19_valence", "video_19_arousal", "video_19_no_ratings", "video_20_valence", "video_20_arousal", "video_20_no_ratings",
                                  "video_21_valence", "video_21_arousal", "video_21_no_ratings", "video_22_valence", "video_22_arousal", "video_22_no_ratings", "video_23_valence", "video_23_arousal", "video_23_no_ratings", "video_25_valence", "video_25_arousal", "video_25_no_ratings", "video_29_valence", "video_29_arousal", "video_29_no_ratings", "video_31_valence", "video_31_arousal", "video_31_no_ratings", "video_33_valence", "video_33_arousal", "video_33_no_ratings", "video_37_valence", "video_37_arousal", "video_37_no_ratings", "video_38_valence", "video_38_arousal", "video_38_no_ratings", "video_39_valence", "video_39_arousal", "video_39_no_ratings",
                                  "video_41_valence", "video_41_arousal", "video_41_no_ratings", "video_42_valence", "video_42_arousal", "video_42_no_ratings", "video_46_valence", "video_46_arousal", "video_46_no_ratings", "video_48_valence", "video_48_arousal", "video_48_no_ratings", "video_49_valence", "video_49_arousal", "video_49_no_ratings", "video_51_valence", "video_51_arousal", "video_51_no_ratings", "video_55_valence", "video_55_arousal", "video_55_no_ratings", "video_56_valence", "video_56_arousal", "video_56_no_ratings", "video_57_valence", "video_57_arousal", "video_57_no_ratings", "video_58_valence", "video_58_arousal", "video_58_no_ratings"]

    return average_av_ratings_columns