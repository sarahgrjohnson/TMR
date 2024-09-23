# Load NWB & remove bad chans

import os
import glob
import mne
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries
from pynwb.misc import TimeSeries
import numpy as np


def load_nwb(file_path):
    with NWBHDF5IO(file_path, 'r') as io:
        nwb = io.read()
        ecog_data = nwb.acquisition['ieeg'].data[:]
        sfreq = nwb.acquisition['ieeg'].rate
        channel_names = nwb.acquisition['ieeg'].electrodes.data[:]
    return nwb, ecog_data, sfreq, channel_names


def plot_channel(raw, channel_idx, ax):
    data = raw.get_data()[channel_idx]
    times = raw.times[:10000]  # Plot first 10000 samples
    ax.plot(times, data[:10000])
    ax.set_title(f"Channel {raw.ch_names[channel_idx]}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")


def remove_channels_by_name(nwb, bad_channel_names):
    electrodes_table = nwb.electrodes
    all_electrode_names = electrodes_table['label'].data[:]
    all_electrode_names = [str(name) for name in all_electrode_names]
    keep_indices = [i for i, name in enumerate(all_electrode_names) if name not in bad_channel_names]

    if len(keep_indices) == len(all_electrode_names):
        print("No bad channels matched. Nothing was removed.")
        return nwb

    print(f"Removing bad channels: {bad_channel_names}")

    # Create a new electrodes table with only the good channels
    new_electrodes = electrodes_table[keep_indices]
    nwb.electrodes.clear_columns()
    for column_name in new_electrodes.colnames:
        nwb.electrodes.add_column(
            name=column_name,
            description=new_electrodes[column_name].description,
            data=new_electrodes[column_name].data
        )

    # Update the ieeg data
    ieeg_series = nwb.acquisition['ieeg']
    filtered_data = ieeg_series.data[:, keep_indices]

    # Create a new ElectricalSeries with the filtered data
    new_ieeg_series = ElectricalSeries(
        name='ieeg',
        data=filtered_data,
        electrodes=nwb.create_electrode_table_region(
            region=list(range(len(keep_indices))),
            description='filtered electrodes'
        ),
        starting_time=ieeg_series.starting_time,
        rate=ieeg_series.rate,
        description=ieeg_series.description
    )

    # Replace the old ieeg series with the new one
    nwb.acquisition['ieeg'] = new_ieeg_series

    return nwb


def remove_outspikeysoz(nwb):
    electrodes_table = nwb.electrodes

    # Find where soz, spikey, and out are not 1
    keep_indices = [i for i, soz, spikey, out in zip(
        range(len(electrodes_table)),
        electrodes_table['soz'].data,
        electrodes_table['spikey'].data,
        electrodes_table['out'].data
    ) if not (soz or spikey or out)]

    nwb.electrodes = electrodes_table[keep_indices]
    ieeg_series = nwb.acquisition['ieeg']
    filtered_data = ieeg_series.data[:, keep_indices]
    ieeg_series.data = filtered_data

    return nwb


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
    print(f"  Input: {nwb_file}")
    print(f"  Prep output: {prep_file}")

    # Load the NWB file
    nwb, ecog_data, sfreq, channel_names = load_nwb(nwb_file)

    # Channel names to a string like mne wants
    channel_names_str = [str(channel) for channel in channel_names]

    # Create MNE Raw object
    info = mne.create_info(ch_names=channel_names_str, sfreq=sfreq, ch_types='ecog')
    raw = mne.io.RawArray(ecog_data.T, info)

    # Input bad chans
    bad_channel_names_input = input("Enter bad channel names (comma-separated): ")
    bad_channels = [name.strip() for name in bad_channel_names_input.split(",")]

    existing_bad_channels = [ch for ch in bad_channels if ch in channel_names_str]
    non_existing_channels = set(bad_channels) - set(existing_bad_channels)

    if non_existing_channels:
        print(f"Warning: The following channels were not found in the NWB file: {', '.join(non_existing_channels)}")

    if existing_bad_channels:
        print(f"Removing the following bad channels: {', '.join(existing_bad_channels)}")
    else:
        print("No valid bad channels to remove... weird.")

    # Remove bad channels and write new NWB file
    with NWBHDF5IO(nwb_file, mode='r') as read_io:
        nwbfile = read_io.read()
        if existing_bad_channels:
            nwbfile = remove_channels_by_name(nwbfile, existing_bad_channels)
            nwbfile.set_modified()

        with NWBHDF5IO(prep_file, mode='w') as export_io:
            export_io.export(src_io=read_io, nwbfile=nwbfile)

    try:
        if bad_channels:
            nwbfile = remove_channels_by_name(nwbfile, bad_channels)
        else:
            print("No bad channels were removed... weird")
    except (TypeError, ValueError) as e:
        print(e)

    # Remove bad chans and write new NWB file
    with NWBHDF5IO(nwb_file, mode='r') as read_io:
        nwbfile = read_io.read()
        nwbfile = remove_channels_by_name(nwbfile, bad_channels)
        nwbfile.set_modified()

        with NWBHDF5IO(prep_file, mode='w') as export_io:
            export_io.export(src_io=read_io, nwbfile=nwbfile)

    print(f"Preprocessed data saved to {prep_file}")
    print(f"Bad channels: {bad_channels}")

print("Processing complete for all sessions.")

# Jupyter notebook widget thingy
# from IPython.core.display_functions import display
# from ipywidgets import interact, interactive, fixed
# from ipywidgets import widgets

# Pick out bad chans
# def bad_chan_select(raw):
#     bad_channels = []
#     n_channels = len(raw.ch_names)
#     fig, axs = pyplt.subplots(n_channels, 1, figsize=(15, 5 * n_channels))
#
#     # Plot each channel
#     for i, ax in enumerate(axs):
#         plot_channel(raw, i, ax)
#     pyplt.show()
#
#     while True:
#         channel_input = input("Enter the number of a bad channel (or 'done' to finish): ")
#         if channel_input.lower() == 'done':
#             break
#         try:
#             channel_num = int(channel_input)
#             if channel_num < 0 or channel_num >= n_channels:
#                 print("Invalid channel number. Please try again.")
#             elif raw.ch_names[channel_num] not in bad_channels:
#                 bad_channels.append(raw.ch_names[channel_num])
#                 print(f"Marked {raw.ch_names[channel_num]} as bad")
#             else:
#                 print(f"{raw.ch_names[channel_num]} already marked as bad")
#         except ValueError:
#             print("Invalid input. Please try again.")
#
#     return bad_channels