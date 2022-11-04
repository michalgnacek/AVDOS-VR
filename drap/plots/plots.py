# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 19:53:38 2022

@author: Michal Gnacek
"""

from ..utils.files_handler import check_or_create_folder
import matplotlib.pyplot as plt

#%%
def __get_plot_save_filepath(plot_directory, file_type, participant_name):
    figure_save_filepath = plot_directory
    if('Slow Movement' in file_type):
        figure_save_filepath = figure_save_filepath + '/slow_movemet_'
    elif('Fast Movement' in file_type):
        figure_save_filepath = figure_save_filepath + '/fast_movemet_'
    elif('Video' in file_type):
        figure_save_filepath = figure_save_filepath + '/video_movemet_'
        if('1' in file_type):
            figure_save_filepath = figure_save_filepath + '1'
        elif('2' in file_type):
            figure_save_filepath = figure_save_filepath + '2'
        elif('3' in file_type):
            figure_save_filepath = figure_save_filepath + '3'
        elif('4' in file_type):
            figure_save_filepath = figure_save_filepath + '4'
        elif('5' in file_type):
            figure_save_filepath = figure_save_filepath + '5'
    return figure_save_filepath + participant_name + '.png'

#%%
def plot_sm_ppg(signal, events, participant_name, root_folder=""):
    df = signal[['Time', 'Ppg/Raw.ppg']]
    plt.figure(figsize=(10,5))
    plt.plot(df['Time'].values,df['Ppg/Raw.ppg'].values)
    plt.xlim([0, df['Time'].tail(1).item()])
    plt.xlabel('Time(seconds)')
    plt.title('Slow movement PPG and Events for' + participant_name)
    plt.vlines(x=events['Time'], ymin=0, ymax=60000, colors='yellow', lw=2, label='vline_multiple - full height')
    path = __get_plot_save_filepath(root_folder+"plots/sm_ppg", "Slow Movement", participant_name)
    check_or_create_folder(path)
    plt.savefig(path)
    plt.close()

#%%
def plot_fit_state(signal, events, participant_name, file_type, root_folder=""):
    df = signal[['Time', 'Faceplate/FitState']]
    plt.figure(figsize=(10,5))
    plt.plot(df['Time'].values,df['Faceplate/FitState'].values)
    plt.xlim([0, df['Time'].tail(1).item()])
    plt.xlabel('Time(seconds)')
    plt.title(file_type + ' - ' + 'Contact FitState for ' + participant_name)
    path1 = __get_plot_save_filepath(root_folder+"plots/contact_state", file_type, participant_name)
    check_or_create_folder(path1)
    plt.savefig(path1)

    plt.title(file_type + ' - ' + 'Contact FitState and Events for ' + participant_name)
    plt.vlines(x=events['Time'], ymin=0, ymax=10, colors='yellow', lw=2, label='vline_multiple - full height')
    path2 = __get_plot_save_filepath(root_folder+"plots/contact_state_and_events", file_type, participant_name)
    check_or_create_folder(path2)
    plt.savefig(path2)
    plt.close()