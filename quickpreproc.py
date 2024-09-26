import os
import glob
import mne
import numpy as np
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.ecephys import ElectricalSeries

# START HERE
data_dir = '/home/hbml/Desktop/Sarah/TMR/RAW/'
subject = 'sub-NS155_02'
sessions = ['preTest', 'postTest', 'overnight']

raw_dir = os.path.join(data_dir, 'Recordings', 'Neural')
prep_base_dir = os.path.join(data_dir, 'Recordings', 'Neural_prep')
os.makedirs(prep_base_dir, exist_ok=True)

for session in sessions:
    nwb_pattern = os.path.join(raw_dir, f'{subject}_ses-implant02_task-TMR_acq-{session}_run-*.0_ieeg.nwb')
    nwb_files = glob.glob(nwb_pattern)

    if not nwb_files:
        print(f"No NWB file found for session {session}")
        print(f" Looking for: {nwb_pattern}")
        print(f" Files in directory: {os.listdir(raw_dir)}")
        continue

    nwb_file = nwb_files[0]
    prep_dir = os.path.join(prep_base_dir, subject, session)
    os.makedirs(prep_dir, exist_ok=True)

    base_name = os.path.basename(nwb_file).replace('.nwb', '')
    prep_file = os.path.join(prep_dir, f'{base_name}_prep.nwb')

    print(f"Processing {session} data:")
    print(f"  Input: {nwb_file}")
    print(f"  Prep output: {prep_file}")

    with NWBHDF5IO(nwb_file, 'r') as io:
        raw_io = NWBHDF5IO(nwb_file, "r")
        nwb = raw_io.read()

        # Load
        ecog_data = nwb.acquisition['ieeg'].data[:]
        sfreq = nwb.acquisition['ieeg'].rate
        channel_names = nwb.acquisition['ieeg'].electrodes.data[:]

        # Channel names to a string like mne wants
        channel_names_str = [str(channel) for channel in channel_names]

        # Create MNE Raw object
        info = mne.create_info(ch_names=channel_names_str, sfreq=sfreq, ch_types='ecog')
        raw = mne.io.RawArray(ecog_data.T, info)

        # Downsample to 500 Hz
        raw.resample(500)
        print("Downsampled to 500 Hz.")

        # Notch filters
        notch_freqs = np.arange(60, 241, 60)  # 60, 120, 180, 240 Hz
        raw.notch_filter(notch_freqs)
        print("Applied notch filters.")

        # Rereferencing
        raw.set_eeg_reference('average')
        print("Done referencing.")

        # Create a processing module for the preprocessed data
        mod = nwb.create_processing_module(
            name='preprocessing',
            description='Preprocessing steps: downsampling, notch filtering, and rereferencing'
        )

        # Create an ElectricalSeries with the preprocessed data
        filt_data = raw.get_data()
        sfreq = np.array([sfreq])
        ts2 = TimeSeries(name="preprocessed_electricalseries", data=filt_data, unit="m", timestamps=sfreq)
        # Add the ElectricalSeries to the processing module
        mod.add_container(ts2)

        # Add preprocessed module to NWB 
        with NWBHDF5IO(prep_file, mode="w", manager=raw_io.manager) as io:
            io.write(nwb)

    print(f"Preprocessed data saved to NWB file: {prep_file}")

print("Preprocessing complete.")