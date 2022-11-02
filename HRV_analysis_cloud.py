import pandas as pd
import numpy as np


from HRV_utils_cloud import butter_bandpass_filter, medianFilter_merged, moving_average, detect_RR, hampel_filtering, winsorize_signal, detect_RR_dynamic, shift_peaks

import nolds


feature_names_time = ['mean_hr','ibi','sdnn','median_ibi','sdsd','rmssd','pnn20','pnn50','median_diff','rr_range','rr_iqrange','cv_rmssd','cv_sdnn']
feature_names_hist = ['tinn','hti']
features_names_poincare= ['sd1','sd2','sd1/sd2','area_ell','csi','cvi','csi_mod','GI','SI','AI','PI','C1d','C1a','SD1d','SD1a','C2d','C2a','SD2d','SD2a','Cd','Ca','SDNNd','SDNNa']

feature_names_complexity = ['samp_ent']


def HRV_complexity(RR_intervals) :
    
    #dataframe to structure the features and feature names
    out_df = pd.DataFrame([np.zeros(len(feature_names_complexity))],columns = feature_names_complexity)
    
    #sample entropy of the data
    out_df['samp_ent'] = float(nolds.sampen(RR_intervals, emb_dim=2))
    median_entropy = 1.4311004404647343 #pre-calculated value from 40 particiopants
    out_df['samp_ent'] = out_df['samp_ent'].replace([np.inf, -np.inf], median_entropy)#avoud inf
    
    return out_df

#extract HRV features in time domain
def HRV_time(RR_intervals):
    #dataframe to structure the features and feature names
    out_df = pd.DataFrame([np.zeros(len(feature_names_time))],columns = feature_names_time)
    
    out_df['mean_hr'] = 60/(np.mean(RR_intervals)/1000)#to seconds and then to BPMs
    out_df['ibi'] = ibi = np.mean(RR_intervals) 
    out_df['median_ibi']  = np.median(RR_intervals) 
    
    out_df['sdnn'] = np.std(RR_intervals)

    RR_diff =[] #difference of successive RRs
    RR_sqdiff = [] # sqrt of the differences
    for i in range(len(RR_intervals)-1):  
        #RR_diff.append(np.absolute(RR_intervals[i+1]-RR_intervals[i]))
        RR_diff.append(RR_intervals[i+1]-RR_intervals[i])
        RR_sqdiff.append(np.power(RR_intervals[i+1]-RR_intervals[i],2))
        #RR_sqdiff.append(np.power(np.absolute(RR_intervals[i+1]-RR_intervals[i]),2)) 
        
    RR_diff=np.array(RR_diff)
    RR_sqdiff = np.array(RR_sqdiff)
    
    out_df['sdsd'] =sdsd = np.std(RR_diff) 
    out_df['rmssd'] = rmssd = np.sqrt(np.mean(RR_sqdiff))
    
    #nn20 =  RR_diff[RR_diff>0.02] #all values over 20 seconds
    #nn50 =  RR_diff[RR_diff>0.05]#all values over 20 seconds
    nn20 =  RR_diff[abs(RR_diff)>20] #all values over 20 ms
    nn50 =  RR_diff[abs(RR_diff)>50]#all values over 50 ms
    out_df['pnn20'] = 100*float(len(nn20)) / float(len(RR_diff))
    out_df['pnn50'] = 100*float(len(nn50)) / float(len(RR_diff))

    out_df['median_diff'] = np.median(RR_diff)
    out_df['rr_range'] = np.max(RR_intervals) - np.min(RR_intervals)
    out_df["rr_iqrange"] = np.percentile(RR_intervals,75)-np.percentile(RR_intervals,25)
    out_df['cv_rmssd'] = rmssd / ibi
    out_df['cv_sdnn'] =  sdsd / ibi

    return out_df




#extract HRV features based on RR histogram
def HRV_hist(RR_intervals):
     #dataframe to structure the features and feature names
    out_df = pd.DataFrame([np.zeros(len(feature_names_hist))],columns = feature_names_hist)
    
    height, bins = np.histogram(RR_intervals, bins="auto")
    out_df['tinn'] = np.max(bins) - np.min(bins)  # Triangular Interpolation of the NN Interval Histogram
    out_df['hti'] = len(RR_intervals) / np.max(height)  # HRV Triangular Index
    
    return out_df

#extract features based on Poincare plot
#From: https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/hrv/hrv_nonlinear.py
#extract HRV features based on RR histogram
def HRV_hist(RR_intervals):
     #dataframe to structure the features and feature names
    out_df = pd.DataFrame([np.zeros(len(feature_names_hist))],columns = feature_names_hist)
    
    height, bins = np.histogram(RR_intervals, bins="auto")
    out_df['tinn'] = np.max(bins) - np.min(bins)  # Triangular Interpolation of the NN Interval Histogram
    out_df['hti'] = len(RR_intervals) / np.max(height)  # HRV Triangular Index
    
    return out_df

#extract features based on Poincare plot
#From: https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/hrv/hrv_nonlinear.py
def HRV_poincare(RR_intervals):
    
    #dataframe to structure the features and feature names
    out_df = pd.DataFrame([np.zeros(len(features_names_poincare))],columns = features_names_poincare)

    #helper variables
    x = np.array(RR_intervals[:-1])
    y = np.array(RR_intervals[1:])
    N = len(RR_intervals) - 1

    sd1 = np.std((x-y) / np.sqrt(2))
    sd2 = np.std((x+y)/ np.sqrt(2))
    out_df['sd1'] = sd1
    out_df['sd2'] = sd2
    out_df['sd1/sd2'] = sd1/sd2

    # Area of ellipse described by SD1 and SD2 (also known as S)
    out_df['area_ell']= np.pi * sd1 * sd2

    # CSI / CVI
    T = 4 * sd1
    L = 4 * sd2
    out_df['csi']= L / T
    out_df['cvi']= np.log10(L * T)
    out_df['csi_mod']= L ** 2 / T

    #x = x*1000 #seconds to ms
    #y = y*1000 #seconds to ms

    diff = y-x
    decelerate_indices = np.where(diff > 0)[0]  # idx where y > x
    accelerate_indices = np.where(diff < 0)[0]  # idx where y < x
    nochange_indices = np.where(diff == 0)[0]

    # Distances to centroid line l2
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    dist_l2_all = abs((x - centroid_x) + (y - centroid_y)) / np.sqrt(2)

    # Distances to LI
    dist_all = abs(y - x) / np.sqrt(2)

    # Calculate the angles
    theta_all = abs(np.arctan(1) - np.arctan(y / x))  # phase angle LI - phase angle of i-th point
    # Calculate the radius
    r = np.sqrt(x ** 2 + y ** 2)
    # Sector areas
    S_all = 1 / 2 * theta_all * r ** 2

    # Guzik's Index (GI)
    den_GI = np.sum(dist_all)
    num_GI = np.sum(dist_all[decelerate_indices])
    out_df["GI"] = (num_GI / den_GI) * 100

    # Slope Index (SI)
    den_SI = np.sum(theta_all)
    num_SI = np.sum(theta_all[decelerate_indices])
    out_df["SI"] = (num_SI / den_SI) * 100
    

    # Area Index (AI)
    den_AI = np.sum(S_all)
    num_AI = np.sum(S_all[decelerate_indices])
    out_df["AI"] = (num_AI / den_AI) * 100
    

    # Porta's Index (PI)
    m = N - len(nochange_indices)  # all points except those on LI
    b = len(accelerate_indices)  # number of points below LI
    out_df["PI"] = (b / m) * 100
   

    # Short-term asymmetry (SD1)
    sd1d = np.sqrt(np.sum(dist_all[decelerate_indices] ** 2) / (N - 1))
    sd1a = np.sqrt(np.sum(dist_all[accelerate_indices] ** 2) / (N - 1))

    sd1I = np.sqrt(sd1d ** 2 + sd1a ** 2)
    out_df["C1d"] = (sd1d / sd1I) ** 2
    out_df["C1a"] = (sd1a / sd1I) ** 2
    out_df["SD1d"] = sd1d  # SD1 deceleration
    out_df["SD1a"] = sd1a  # SD1 acceleration
   

    # Long-term asymmetry (SD2)
    longterm_dec = np.sum(dist_l2_all[decelerate_indices] ** 2) / (N - 1)
    longterm_acc = np.sum(dist_l2_all[accelerate_indices] ** 2) / (N - 1)
    longterm_nodiff = np.sum(dist_l2_all[nochange_indices] ** 2) / (N - 1)

    sd2d = np.sqrt(longterm_dec + 0.5 * longterm_nodiff)
    sd2a = np.sqrt(longterm_acc + 0.5 * longterm_nodiff)

    sd2I = np.sqrt(sd2d ** 2 + sd2a ** 2)
    out_df["C2d"] = (sd2d / sd2I) ** 2
    out_df["C2a"] = (sd2a / sd2I) ** 2
    out_df["SD2d"] = sd2d  # SD2 deceleration
    out_df["SD2a"] = sd2a  # SD2 acceleration
   

    # Total asymmerty (SDNN)
    sdnnd = np.sqrt(0.5 * (sd1d ** 2 + sd2d ** 2))  # SDNN deceleration
    sdnna = np.sqrt(0.5 * (sd1a ** 2 + sd2a ** 2))  # SDNN acceleration
    sdnn = np.sqrt(sdnnd ** 2 + sdnna ** 2)  # should be similar to sdnn in hrv_time
    out_df["Cd"] = (sdnnd / sdnn) ** 2
    out_df["Ca"] = (sdnna / sdnn) ** 2
    out_df["SDNNd"] = sdnnd
    out_df["SDNNa"] = sdnna
    

    return out_df



#filter signal and calculate HRV features
def get_HRV_features(_sample,ma=False,detrend=False,band_pass=False, thres=.75,winsorize=True,
                     dynamic_threshold=True,dynamic_threshold_value=1.5,
                     winsorize_value=25,hampel_fiter=True,sampling=1000):
 
    sample = _sample.copy()
    if band_pass:
        sample = butter_bandpass_filter(sample, fs=sampling)
    if ma:
        ma_size = 10
        sample = moving_average(sample,ma_size = ma_size)
    #removes outliers above and below the threshold (in percentiles)
    if winsorize:
        sample=winsorize_signal(sample,winsorize_value)
    if dynamic_threshold: #find the median of the min-max normalized signal
        thres = dynamic_threshold_value*np.median((sample - sample.min())/(sample.max() - sample.min()))
   
    #rr,timings,peak_indx =  detect_RR(sample,thres,sampling)
    
    rr,timings,peak_indx =  detect_RR_dynamic(sample,thres,sampling)

   

    if len(rr)<len(sample)/(2*sampling): #check whether HR is < 30
        print("Bad signal. Too little RRs detected.")
        return np.array([-1]*len(feature_names_time)),sample,rr,timings,peak_indx
    elif len(rr)>len(sample)/(sampling/4):  #check whether HR is > 240
        print("Bad signal. Too much RRs detected.")
        return np.array([-1]*len(feature_names_time)),sample,rr,timings,peak_indx
    
    #remove shift caoused by filters
    peak_ind_original = peak_indx
    if ma:
        peak_ind_original = peak_indx+ma_size-2        #the signal is already shigted by the moving average
    shift_sec = (np.median(rr)/1000)/6#to seconds then devided by 6
    rr,timings,peak_indx = shift_peaks(peak_ind_original,_sample,shift_sec =shift_sec , sampling_rate=sampling)

    #smooths outliers with median values of its neighbours
    if hampel_fiter:
        rr,outlier_indices = hampel_filtering(rr)
    
    hrv_time_features = HRV_time(rr)
    hRV_hist_features = HRV_hist(rr)
    hrv_poincare_features = HRV_poincare(rr)
    hrv_complexity_features = HRV_complexity(rr)

    hrv_features = np.concatenate([hrv_time_features.iloc[0,:],hRV_hist_features.iloc[0,:],hrv_poincare_features.iloc[0,:],hrv_complexity_features.iloc[0,:]])
    
    return hrv_features,sample,rr,timings,peak_indx
