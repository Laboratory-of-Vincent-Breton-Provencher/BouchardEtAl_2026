#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:17:17 2022
@author: vbp

Figure 1B: region-specific dopamine dynamics (mPFC, BLA, NAc_lat, DS)
aligned to reward delivery, with lick raster.

LOAD DATA FROM
'4 fibers all data/DA4fib02/20240117_naiverec/'
(new fiber photometry setup)
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# %% Global plotting style
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.linewidth'] = 0.5
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['xtick.major.width'] = 0.5

plt.close('all')

# %% Parameters

# Path to the example session data. This session needed a manual t0 offset
# to correct for a TTL acquisition issue.
F = 'Data/ExampleSession/'
t0 = 35.9

# Region order in which subplots are drawn (indices refer to columns 3-6
# of the FP data, in the order [mPFC, NAc_lat, DS, BLA]).
ls_reg_names = ['mPFC', 'NAc_lat', 'DS', 'BLA']
ls_col = ['#400D60', '#F3703A', '#ED1651', '#0055A1']
ls_ylim = [[-0.25, 0.4], [-0.5, 1.5], [-2.5, 5], [-0.1, 0.25]]
ls_reg = [0, 3, 1, 2]  # display order: mPFC, BLA, NAc_lat, DS


# %% Functions

def butter_highpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='high', analog=False)


def butter_highpass_filter(data, cutoff, fs, order=2):
    b, a = butter_highpass(cutoff, fs, order=order)
    return lfilter(b, a, data)


# %% Load session files

listFiles = [f for f in os.listdir(F) if os.path.isfile(os.path.join(F, f))]


def find_single_file(keyword, files=listFiles, folder=F):
    """Return the path to the unique file containing `keyword`, or exit."""
    matches = [f for f in files if keyword in f]
    if len(matches) > 1:
        print(f"There is more than one '{keyword}' file in {folder}")
        sys.exit()
    if len(matches) == 0:
        print(f"No file containing '{keyword}' found in {folder}")
        sys.exit()
    return os.path.join(folder, matches[0])


#### Load params
with open(find_single_file('params'), 'r') as f:
    params = json.loads(f.read())

#### Load trial matrix
trmtx = pd.read_csv(find_single_file('trialMTX'))

#### Load TTLs
ard_path = find_single_file('ard2')
ard = np.genfromtxt(ard_path, delimiter=',', dtype='str')

# Keep only the last 7 characters of each line, then drop malformed
# (non-numeric) lines, e.g. from a broken serial read.
ard = np.array([x[-7:] for x in ard])
ard = ard[np.char.isdigit(ard)]

# Parse TTL lines into a table: columns 0-2 are TTL channels, x is the
# running sample counter.
TTLs = np.zeros((ard.shape[0], 3))
x = np.zeros(ard.shape[0])
for k, a in enumerate(ard):
    if len(a) > 6:
        TTLs[k, 0] = int(a[0])
        TTLs[k, 1] = int(a[1])
        TTLs[k, 2] = int(a[2])
        x[k] = int(a[3:])
    else:
        x[k] = -1

# Remove arduino reads before acquisition start in Bonsai
TTLs = TTLs[np.where(x == 0)[0][0]:, :]

#### Load photometry data and time stamps
fp_path = find_single_file('FP2')
FP = np.genfromtxt(fp_path, delimiter=',', dtype='float', skip_header=1)

# Adjust TTLs and FP length if they're off by a sample or two
if FP.shape[0] != TTLs.shape[0]:
    if abs(FP.shape[0] - TTLs.shape[0]) <= 2:
        idLast = min(FP.shape[0], TTLs.shape[0])
        FP = FP[:idLast, :]
        TTLs = TTLs[:idLast]
    else:
        print(f"There is a huge mismatch between TTLs (n={TTLs.shape[0]}) "
              f"and FP data (n={FP.shape[0]}). Parsing of Ard might need to be fixed")
        sys.exit()

# Timestamps from camera, keeping only frames flagged for the green channel
ts = (FP[:, 0] - FP[0, 0]) / 1e9
ts_dff = ts[TTLs[:, 0] == 1]


# %% Plot dopamine traces per region, aligned to reward delivery

fig, ax = plt.subplots(5, sharex=True, figsize=(8, 5))

for ii in range(4):
    r = ls_reg[ii]

    # Extract raw fluorescence for this region (columns start at index 2)
    fraw = FP[:, r + 2]

    # Split into the two interleaved channels (signal / isosbestic)
    f1 = fraw[TTLs[:, 0] == 1]
    f2 = fraw[TTLs[:, 1] == 1]

    # Trim to matching length in case one channel has an extra sample
    nts = min(f1.shape[0], f2.shape[0])
    f1, f2, ts_dff = f1[:nts], f2[:nts], ts_dff[:nts]

    # Sampling rate from the actual timestamps
    FPS = 1 / np.mean(np.diff(ts_dff))

    # High-pass filter both channels, then subtract isosbestic from signal
    f1_filt = butter_highpass_filter(f1 - np.mean(f1), 0.005, FPS, order=2)
    f2_filt = butter_highpass_filter(f2 - np.mean(f2), 0.005, FPS, order=2)
    df = f1_filt - f2_filt

    # Light smoothing (5-point moving average)
    df = np.convolve(df, np.ones(5) / 5, mode='same')

    # Align to reward delivery: tone onset time + t0 offset + delay to reward
    t_reinf = trmtx['timeTone'] + t0 + 1.5

    idx = np.zeros(len(t_reinf))
    for i in range(len(t_reinf)):
        idx[i] = np.where(ts_dff >= t_reinf[i])[0][0]
    idx = idx[trmtx['Reward?'] > 0].astype(int)

    a = ax[ii]
    a.plot(ts_dff, df, color='k')
    yl = a.get_ylim()
    a.vlines(ts_dff[idx], *yl, ls=':', color='k', lw=1)
    a.set_ylim(ls_ylim[r])
    a.set_ylabel(ls_reg_names[r])


# %% Load arduino lick data and add lick raster

ardLick_path = find_single_file('_ArdData')
ardLick = pd.read_csv(ardLick_path)
t_ard = ardLick['# TimeMATLAB'] + t0

# Detect lick onsets (rising edges) for the event plot
lick_onset = np.diff(ardLick[' LICK1'], append=0) > 0
t_lick = t_ard[lick_onset]

a = ax[4]
a.eventplot(t_lick, orientation='horizontal', color='k', lw=0.5)
yl = a.get_ylim()
a.vlines(ts_dff[idx], *yl, ls=':', color='k', lw=1)

a.set_xlim([100, 151])
a.set_ylim([0.4, 1.6])
a.set_xlabel('Time (s)')
a.set_ylabel('Lick')

fig.tight_layout()

# Uncomment to save the figure:
# fig.savefig('Fig1B.pdf')