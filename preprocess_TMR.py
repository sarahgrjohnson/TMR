"""

Max's movie preprocessing scripts (@maxne)
Adapted for minimal preprocessing of TMR data by Sarah

TO DO:
- Align TTL pulses

"""

import os, sys
import glob
import numpy as np
import mne
from pynwb import NWBHDF5IO
import matplotlib

sys.path.append('/home/hbml/Documents/GitHub/EPIPE_20231005/Python')
from epipe import inspectNwb, nwb2mne, read_ielvis, reref_avg, reref_bipolar, filter_hfa_continuous

matplotlib.use('Qt5Agg')

# Files to preprocess
subject = 'NS155_02'
sessions = ['preTest', 'postTest', 'overnight']

# Set directories
data_dir = '/home/hbml/Desktop/Sarah/TMR/'
raw_dir = os.path.join(data_dir, 'RAW')
prep_base_dir = os.path.join(data_dir, 'Prep')
ttl_dir = os.path.join(data_dir, 'Beh')
anat_dir = '/home/hbml/freesurfer/subjects/'
fs_dir = os.path.join(anat_dir, subject)
os.makedirs(prep_base_dir, exist_ok=True)

# Parameters
resample_fs = 500
notch_freqs = [60, 120, 180, 240]
ref_types = ['avg']

# Load data

for session in sessions:
    nwb_pattern = os.path.join(raw_dir, f'sub-{subject}_ses-implant0*_task-TMR_acq-{session}_run-*.0_ieeg.nwb')
    nwb_files = glob.glob(nwb_pattern)

    nwb_file = nwb_files[0]
    prep_dir = os.path.join(prep_base_dir, subject, session)
    os.makedirs(prep_dir, exist_ok=True)

    # Read NWB
    if nwb_file:
        nwb_file = nwb_files[0]
        io = NWBHDF5IO(nwb_file, mode='r', load_namespaces=True)
        nwb = io.read()
    else:
        print(f"No NWB file found for session {session}")
        print(f" Looking for: {nwb_pattern}")
        print(f" Files in directory: {os.listdir(raw_dir)}")
        continue

    # Get info
    nwbInfo = inspectNwb(nwb)
    tsInfo = nwbInfo['timeseries']
    elecTable = nwbInfo['elecs']

    # Get iEEG data
    if 'ieeg' in tsInfo['name'].to_list():
        ecogContainer = nwb.acquisition.get('ieeg')
        fs = ecogContainer.rate
        ecog = nwb2mne(ecogContainer, preload=False)

        # Get coordinates of each electrode that has
        ielvis_df = read_ielvis(fs_dir)
        ch_coords = {}
        nan_array = np.empty((3,)) * np.nan
        for thisChn in ecog.ch_names:
            idx = np.where(ielvis_df['label'] == thisChn)[0]
            if len(idx) == 1:
                xyz = np.array(ielvis_df.iloc[idx[0]]['LEPTO'])
                ch_coords[thisChn] = xyz / 1000
            elif len(idx) == 0:
                ch_coords[thisChn] = nan_array
            else:
                raise ValueError('More than 1 found!')

        # Create `montage` data structure as required by MNE
        montage = mne.channels.make_dig_montage(ch_pos=ch_coords, coord_frame='mri')
        montage.add_estimated_fiducials(subject, anat_dir)
        ecog.set_montage(montage)

    # Get the current sampling rate. Important for later
    orig_fs = ecog.info['sfreq']

    # Get the TTL pulses. Specify the name of the container with the TTL pulses
    ttl_container_name = 'TTL'
    try:
        ttls = nwb.get_acquisition(ttl_container_name).timestamps[()]
    except:
        # An analog TTL channel, convert to discrete timestamps
        ana_ttls = nwb.get_acquisition(ttl_container_name).data[()].flatten()
        ttl_rate = nwb.get_acquisition(ttl_container_name).rate
        from epipe import ana2dig

        _, ttls = ana2dig(ana_ttls, fs=ttl_rate, min_diff=0.4, return_time=True)

    ttl_id = nwb.acquisition['TTL'].data[:]

    # Close NWB file
    io.close()

    # Start preprocessing
    preproc_filename = os.path.join(prep_dir, f'sub-{subject}_ses-implant01_task-TMR_acq-{session}_run-01_ieeg_prep.fif')

    if not os.path.exists(preproc_filename):

        print('--->Applying notch filters and downsampling to %2.fHz' % resample_fs)

        # Copy the `ecog` variable and then resample and apply notch filter
        ecogPreproc = ecog.resample(resample_fs).notch_filter(notch_freqs, notch_widths=2)

        # Display the raw traces and mark bad channels
        nbadOrig = ecogPreproc.info['bads']
        fig = ecogPreproc.plot(show=True, block=True, remove_dc=True, duration=15.0, n_channels=16)

        # Save the current state of the data in the MNE format

        ecogPreproc.save(preproc_filename,
                         fmt='single', overwrite=True)

    else:
        ecogPreproc = mne.io.read_raw(preproc_filename, preload=True)

    for ref in ref_types:

        print('#' * 50)
        print('Beginning processing for data using %s reference' % ref)
        print('#' * 50)

        # What the preprocessed filename for this reference type should be
        preprocRerefFname = os.path.join(prep_dir, f'sub-{subject}_ses-implant01_task-TMR_acq-{session}_run-01_ieeg_prep_ref_{ref}.fif')

        # Check if the preprocessed file already exists so you don't have to redo rereferncing functions
        if os.path.isfile(preprocRerefFname):
            ecogReref = mne.io.read_raw_fif(preprocRerefFname, preload=True)
            if 'ecogPreproc' in locals():
                del ecogPreproc
        else:
            if 'ecogPreproc' not in locals():
                ecogPreproc = mne.io.read_raw_fif(preproc_filename, preload=True)

            if ref == 'avg':
                ecogReref = reref_avg(ecogPreproc)

            elif ref == 'bip':
                ecogReref = reref_bipolar(ecogPreproc)

            # Save the referenced data in MNE format
            ecogReref.save(preprocRerefFname, fmt='single', overwrite=True)
            del ecogPreproc
    print(f"Done preprocessing --> {session}")

print("Preprocessing complete for all sessions.")