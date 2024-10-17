"""
Ripple Detection for TMR
Placeholder/work in progress until 

"""

import os
import glob
import mne
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
from LFP_modified import detect_ripple_evs
import pandas as pd


def main():

    # Start up parameters
    parent_dir = '/home/hbml/Desktop/Sarah/TMR/'
    prep_base_dir = os.path.join(parent_dir, 'Prep')
    params_analysis_dir = os.path.join(parent_dir, 'Params')
    subject_tasks = {
        'NS155_02': ['pretest', 'posttest'],
    #    'NS196': ['overnight', 'posttest']
    }
    
    # Filter settings for ripple and IED freq range (from Liz's scripts)
    params = {
        'ripple_band': [80, 140], # Hz, consider altering to 120
        'ripple_forder': 100, # in samples, was 330
        'IED_frequency_band': [25, 60] # in Hz, forder automatically set by PT
    }

    # Liz's Ripple detection parameters
    minDistance = 0.200;            # in sec {was 0.150} % Vaz et al 2019 used 15ms
    minRippleDuration = 0.021;      # in sec {was 0.020}
    maxRippleDuration = 0.250;      # in sec
    th = [2, 3];                     # Ripple detection thresholds (stdev) [onset/offset, peak] {was [2 3]}

    origsfreq = 500;

    # Loop through subjects and tasks
    for subject, tasks in subject_tasks.items():
        
        # Read the CSV file
        csv_file = os.path.join(params_analysis_dir, 'elecs4analysis.csv')
        df = pd.read_csv(csv_file)

        # Get current subject & get target & wm elecs
        subject_df = df[df['subject'] == subject]
        target_electrodes = subject_df[subject_df['elec_type'] == 'target']['label'].tolist()
        wm_electrodes = subject_df[subject_df['elec_type'] == 'wm']['label'].tolist()
        alltargetelecs = [str(ele) for ele in target_electrodes]
        allwmelecs = [str(ele) for ele in wm_electrodes]

        for task in tasks:
            task_dir = os.path.join(prep_base_dir, subject, task)
            fif_pattern = os.path.join(task_dir, f'*_prep_ref_ieeg.fif')
            fif_files = glob.glob(fif_pattern)
            
            if not fif_files:
                print(f"No matching files found in: {task_dir}")
                continue
            
            fif_file = fif_files[0] 
            print(f"Processing file: {fif_file}")
            
            mne_data = mne.io.read_raw_fif(fif_file, preload=True)
            Fs = mne_data.info['sfreq']

            for eleci in range(len(alltargetelecs)):
                try:
                    # Set electrodes
                    ca1_elec = [alltargetelecs[eleci], allwmelecs[eleci]]
                    ca1_ind = [raw.ch_names.index(ca1_elec[0]), raw.ch_names.index(ca1_elec[1])]
                    
                    # Create a bipolar montage
                    raw_bipolar = mne.set_bipolar_reference(raw, anode=ca1_elec[0], cathode=ca1_elec[1])
                    bp_raw = raw_bipolar.get_data()
                    
                    # Ripple and IED filter parameters 
                    ripple_band = [80, 140]  # Ripple bandpass range in Hz
                    IED_band = [25, 60]       # IED frequency range in Hz
                    
                    # Bandpass filter for ripple detection
                    ripple_signal = mne.filter.filter_data(bp_raw[0], Fs, ripple_band[0], ripple_band[1], method='fir')
                    
                    # Bandpass filter for IED detection
                    IED_signal = mne.filter.filter_data(bp_raw[0], Fs, IED_band[0], IED_band[1], method='fir')
            
                    # Common average reference signal
                    car_signal = np.mean(raw.get_data(), axis=0)
            
                    # Low-pass filter to smooth the ripple-band envelope
                    LPcutoff = round(np.mean(ripple_band)/np.pi) # Cutoff frequency for low-pass filter
                    b, a = butter(4, LPcutoff / (Fs / 2), btype='low')
                    ripple_smoothed = filtfilt(b, a, ripple_signal)
            
                    all_signals = [ripple_signal, car_signal, IED_signal]
            
                except Exception as e:
                    print(f"Error processing electrode {ca1_elec}: {e}")
                    raw = mne.io.read_raw_fif(fif_file, preload=True)
                except:
                    print(f"Error reading file: {fif_file}")
                    continue
            
            # Detect ripple events
            ripple_events = detect_ripple_evs(ripple_smoothed, Fs, minDistance, minRippleDuration, maxRippleDuration, th)
            