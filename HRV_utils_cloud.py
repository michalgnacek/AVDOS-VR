import scipy.signal as _signal
import peakutils
import pandas as pd
import numpy as np

def butter_bandpass(l, h, fs, order):
    nyq = 0.5 * fs
    low = l / nyq
    high = h / nyq
    b, a = _signal.butter(order, [low, high], btype='band')
    return b, a
    
def butter_bandpass_filter(data, l=0.5, h=2.75, fs=1000, order=3):
    b, a = butter_bandpass(l, h, fs, order=order)
    y = _signal.filtfilt(b, a, data)
    return pd.Series(y[0:])
    
#remove rr intervals that are too far from the median(rr) - probably due to noise
def medianFilter_merged(rr):
    percentageBorder = 0.8
    median = np.median(rr)
    f_rr = [] #filtered rrs
    f_time=[] # timestamps for each rr in f_rr
    f_rr = rr[(rr/median>=percentageBorder)  & (rr/median<=(2-percentageBorder))]
    current_time=0.0
    for rr in f_rr: #calulate realtive timestamps
        current_time = current_time+rr
        f_time.append(current_time)
    f_time = np.array(f_time)   
    return f_rr,f_time


#perform moving average
def moving_average(sample,ma_size = 10):
    sample = pd.Series(sample)
    sample_ma = sample.rolling(ma_size).mean()
    sample_ma = sample_ma.iloc[ma_size:].values #remove nan values
    return sample_ma

#find peaks for a given PPG singal
#[-1] represents error
#compared to detect_RR(), the parameter min_dist is changed dynamically based on estimated HR
def detect_RR_dynamic(sig,thres,sampling_rate):
    min_dist = sampling_rate/2.75
    peak_indx = peakutils.indexes(sig, thres=thres, min_dist=min_dist)
    time=np.arange(len(sig))
    
    if len(peak_indx) != 0:
        tmp= time[peak_indx]
        timings1 = tmp[0:]
        timings = tmp[1:]
        RR_intervals = timings-timings1[:len(timings1)-1]
        
        median_ibi = np.median(RR_intervals/sampling_rate)
        delta = .6
        #print('initial dist',min_dist)

        min_dist = delta * (sampling_rate*median_ibi)
        #print('dynamic min_dist',min_dist)
        
        peak_indx = peakutils.indexes(sig, thres=thres, min_dist=min_dist)
        time=np.arange(len(sig))
        tmp= time[peak_indx]
        timings1 = tmp[0:]
        timings = tmp[1:]
        RR_intervals = timings-timings1[:len(timings1)-1]
            
        #return RR_intervals/sampling_rate,timings/sampling_rate,peak_indx #seconds
        return (1000*RR_intervals)/sampling_rate,(1000*timings)/sampling_rate,peak_indx #miliseconds
    else:
        return [-1], [-1], [-1]

#find peaks for a given PPG singal
#[-1] represents error
def detect_RR(sig,thres,sampling_rate):
    peak_indx = peakutils.indexes(sig, thres=thres, min_dist=sampling_rate/2.75)
    time=np.arange(len(sig))
    
    if len(peak_indx) != 0:
        tmp= time[peak_indx]
        timings1 = tmp[0:]
        timings = tmp[1:]
        RR_intervals = timings-timings1[:len(timings1)-1]
         
        return RR_intervals/sampling_rate,timings/sampling_rate,peak_indx
    else:
        return [-1], [-1], [-1]
    
#https://www.mathworks.com/help/signal/ref/hampel.html
#compute median and standard deviation 
#of a window using the sample and its six surrounding samples
#If a sample differs from the median by more than three standard deviations, 
#it is replaced with the median. 
#return filtered RRs and outlier indices 
def hampel_filtering(sample_rr):
    outlier_indices = []
    for i in range(len(sample_rr)):
        start = i-3
        end = i+3
        if start<0: #for the first 3 samples calculate median and std using the closest 6 samples
            start=0
            end = end+3-i
        if end>len(sample_rr)-1: #for the last 3 samples calculate median and std using the first 6 samples
            start = len(sample_rr)-7
            end=len(sample_rr)-1

        sample_med = np.median(sample_rr[start:end])
        sample_std = np.std(sample_rr[start:end])
        if abs(sample_rr[i]-sample_med)>3*sample_std:
            sample_rr[i] = sample_med
            outlier_indices.append(i)
        
    return sample_rr,outlier_indices

#remove outlies above and below given percentiles
def winsorize_signal(sample,winsorize_value):
    
    p_min = np.percentile(sample,winsorize_value)
    p_max = np.percentile(sample,100-winsorize_value)
    
    sample[sample>p_max]=p_max
    sample[sample<p_min]=p_min
    
    return sample

#for each peak p (index of R peak), check in the original signal if there is a higher peak close to p
def shift_peaks(peaks, ppg_signal, shift_sec = .01, sampling_rate=100):
    ppg_signal = np.array(ppg_signal)
    timings = peaks.astype(int)
    shift_samples = (sampling_rate//6)+int(shift_sec*sampling_rate)
    new_peaks = []
    for t in peaks:
        start = t-shift_samples
        if start<0:
            start=0
        end = t+shift_samples
        if end>len(ppg_signal)-1:
            end = len(ppg_signal)-1
        if start>=end:
            break
        shift_signal = ppg_signal[start:end] # get a subsample of the signal [t-shift_sec t+shift_sec]
 
        new_peak_relative = np.argmax(shift_signal)  #relative index of new peak
        old_peak_relative = shift_samples #relative index of original peak
        #from relative index to original index
        new_peak = start+new_peak_relative
        new_peaks.append(new_peak)
        #print(t,new_peak_relative,new_peak,start,end)
        #print(shift_signal)
        
    time=np.arange(len(ppg_signal))
    tmp= time[new_peaks]
    timings1 = tmp[0:]
    timings = tmp[1:]
    RR_intervals = timings-timings1[:len(timings1)-1]
        
    return (1000*RR_intervals)/sampling_rate,(1000*timings)/sampling_rate,new_peaks #miliseconds