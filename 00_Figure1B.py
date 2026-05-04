#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:17:17 2022

@author: vbp

LOAD DATA FROM
'4 fibers all data/DA4fib02/20240117_naiverec/'
"""

# Open data from a behavior session and convert to table and save
# This script is for the new fiber photometry set up

import numpy as np
import os
import matplotlib.pyplot as plt
# import vbp
# import scipy.stats as stats
import json
import sys
import pandas as pd
import tkinter as tk
from tkinter import filedialog

from scipy.signal import butter, lfilter, freqz

# %% To change all graphs
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.linewidth'] = 0.5
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['xtick.major.width'] = 0.5

plt.close('all')

# PARAMETERS
w = np.array([-5, 15])
BLFluo = 0.0274 #This value is calculated with fibers plugged in but no LED on. It represents background camera noise

    
# FUNCTIONS
def makeRaster(a,idx,w):
    
    # If window for raster for last idx is larger than a pad a with nans
    if idx[-1] + w[1] > a.shape[0]:
        nMissing = int(idx[-1] + w[1] - a.shape[0] + 1)
        a = np.append(a,np.ones(nMissing)*np.nan)

    # Create a list of index to be used for the raster
    X = np.tile(np.arange(np.diff(w))+w[0],(len(idx),1));
    Y = X.transpose() + idx
    Y = Y.transpose()
    Y = Y.flatten()
    Y = Y.astype(int)
    
    # Select data to plot as raster
    R = a[Y]
    
    # Reshape into a raster
    R = R.reshape((X.shape[0],X.shape[1]))
    
    return R

def butter_highpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='high', analog=False)

def butter_highpass_filter(data, cutoff, fs, order=2):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# SCRIPTS

# Load folder
# root = tk.Tk()
# root.withdraw()
# root.update()
# print('Please select a directory')
# F = filedialog.askdirectory(initialdir = os.getcwd())
# root.update()


#CHANGER LE T0 pour chaque ANIMAL

# F = '/Users/vbp/Library/CloudStorage/OneDrive-UniversitéLaval/00_Project/DA_Reinf_Spatiotemporal/FP_4Fibers_FirstRec/DA4FIB02/20240117_Naiverec/'
F = '/Users/vbp/Library/CloudStorage/OneDrive-UniversitéLaval/00_Project/DA_Reinf_Spatiotemporal/Data_clean/ExampleRawTraceFig1/DA4fib02/20240117_naiverec/'
t0 = 35.9

# Determine list of files
listFiles = [f for f in os.listdir(F) if os.path.isfile(os.path.join(F, f))]

#### Load params
# Check if more than one file with params
fname = [s for s in listFiles if 'params' in s]
if len(fname) > 1:
    print('There is more than one ''params'' file in '+os.path.sep+F)
    sys.exit()
# Load params from folder
f = open(F+os.path.sep+fname[0],'r')
json_obj = f.read()
params = json.loads(json_obj)
f.close()

#### Load trial matrix
# Check if more than one file with trialMTX
fname = [s for s in listFiles if 'trialMTX' in s]
fnameTrMTX = fname # will be used for save name
if len(fname) > 1:
    print('There is more than one trialMTX file in '+os.path.sep+F)
    sys.exit()
# Load trialMTX from folder
trmtx = pd.read_csv(F+os.path.sep+fname[0])

#### Load TTLs
fname = [s for s in listFiles if 'ard2' in s]
if len(fname) > 1:
    print('There is more than one ard2***.csv file in '+os.path.sep+F)
    sys.exit()
ard = np.genfromtxt(F+os.path.sep+fname[0], delimiter = ',', dtype = 'str')

# Clean up lines that are too long
l = np.array([len(x) for x in ard])

# Clean up lines that are too long
ard = np.array([x[-7:] for x in ard])

# Clean up ard for case where there is a broken line 
ard = ard[np.char.isdigit(ard)]

# Parse TTLs in table
k = 0
TTLs = np.zeros((ard.shape[0],3))
x = np.zeros(ard.shape[0])
for a in ard:
    if len(a) > 6:
        TTLs[k,0] = int(a[0])
        TTLs[k,1] = int(a[1])
        TTLs[k,2] = int(a[2])
        x[k] = int(a[3:])
    else:
        x[k] = -1
    k += 1

# Remove arduino reads before aquisition start in Bonsai
TTLs = TTLs[np.where(x == 0)[0][0]:,:]



### Load photometry data and time stamps
fname = [s for s in listFiles if 'FP2' in s]
if len(fname) > 1:
    print('There is more than one FP2***.csv file in '+os.path.sep+F)
    sys.exit()
FP = np.genfromtxt(F+os.path.sep+fname[0], delimiter = ',', dtype = 'float', skip_header = 1)

# Adjust TTLs and FP length
if FP.shape[0] != TTLs.shape[0]:
    if np.abs(FP.shape[0] - TTLs.shape[0]) <= 2:
        idLast = min([FP.shape[0],TTLs.shape[0]])
        FP = FP[:idLast,:]
        TTLs = TTLs[:idLast]
    else:
        print('There is a huge mismastch between TTLs (n='+str(TTLs.shape[0])+') and FP data (n='+str(FP.shape[0])+'). Parsing of Ard might need to be fixed')
        sys.exit()
# Extract ts from camera
ts = (FP[:,0] - FP[0,0])/10**9
ts_dff = ts[TTLs[:,0] == 1] # Take only ts for green channel

nRegion = FP.shape[1]-2
ls_ylim = [[-0.25,0.4],
           [-0.5,1.5],
           [-2.5,5],
           [-0.1,0.25]]
ls_col = ['#400D60','#F3703A','#ED1651','#0055A1',]
ls_reg_names = ['mPFC','NAc_lat','DS','BLA']
# mPFC,NAc_lat,DS,BLA
ls_reg = [0,3,1,2]
fig,ax = plt.subplots(5,sharex=True,figsize=(8,5))
for ii in range(4):
    r = ls_reg[ii]
    
    # Extract fluo data
    fraw = FP[:,r+2]
    
    # Split fluorescence data for each region
    f1 = fraw[TTLs[:,0] == 1]
    f2 = fraw[TTLs[:,1] == 1]
    
    # Control for case where there is an extra time point for one of the two chan
    nts = np.min([f1.shape[0],f2.shape[0]])
    f1 = f1[:nts]
    f2 = f2[:nts]
    ts_dff = ts_dff[:nts]
    
    # Calculate frame per second
    FPS = 1/np.mean(np.diff(ts_dff))
    
    # Filter signal
    f1_filt = butter_highpass_filter(f1-np.mean(f1), 0.005, FPS, order=2)
    f2_filt = butter_highpass_filter(f2-np.mean(f2), 0.005, FPS, order=2)
    
    # Calculate correct signal for isobestic (Green-Iso)
    df = f1_filt - f2_filt
    
    # 5 point average signal
    # f1_filt = np.convolve(f1_filt,np.ones(5)/5,mode='same')
    # f2_filt = np.convolve(f2_filt,np.ones(5)/5,mode='same')
    df = np.convolve(df,np.ones(5)/5,mode='same')
    
    
    # ### Align data to behavior ############
    
    # # Caclulate timing of tone onset using ttl (trial end) and durConsumption and durPreReinf parameters
    t_reinf = trmtx['timeTone'] + t0 + 1.5
    
    # Find idx of timing of tone onset
    idx = np.zeros(len(t_reinf))
    for i in range(len(t_reinf)):
        idx[i] = np.where(ts_dff >= t_reinf[i])[0][0]
    
    idx = idx[trmtx['Reward?'] > 0].astype(int)
    
    a = ax[ii]
    a.plot(ts_dff,df,color=ls_col[r])
    yl = a.get_ylim()
    a.vlines(ts_dff[idx],*yl,ls=':',color = 'k',lw=1)
    
    a.set_ylim(ls_ylim[r])
    a.set_ylabel(ls_reg_names[r])
    
    
#%% Loard arduino data for licks

# Check if more than one file with trialMTX
fname = [s for s in listFiles if '_ArdData' in s]
if len(fname) > 1:
    print('There is more than one arddata file in '+os.path.sep+F)
    sys.exit()

ard = pd.read_csv(F+fname[0])
t_ard = ard['# TimeMATLAB'] + t0

# Convert lick for eventplot
lick = ard[' LICK1']
lick = np.diff(lick, append = 0) > 0
t_lick = t_ard[lick]

# Smooth lick for lick rate trace
# from scipy.ndimage import gaussian_filter
lick = lick.astype(float)
lick = np.convolve(lick,np.ones(100)/100,mode='same')

# fig,ax = plt.subplots(2,sharex=True)
a = ax[4]
a.eventplot(t_lick,orientation='horizontal',color='k',lw=0.5)
yl = a.get_ylim()
a.vlines(ts_dff[idx],*yl,ls=':',color = 'k',lw=1)

a.set_xlim([100,151])
a.set_ylim([0.4,1.6])
a.set_xlabel('Time (s)')
a.set_ylabel('Lick')

# a = ax[1]
# a.plot(t_ard,lick)

fig.tight_layout()
        