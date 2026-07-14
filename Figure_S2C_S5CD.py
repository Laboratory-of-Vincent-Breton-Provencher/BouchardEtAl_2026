#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 20:42:40 2025

@author: vbp

Session-averaged raster plots and amplitude summary for:
    Fig S2C - raster grid, 4 main regions (DS, NAc_lat, BLA, mPFC) x 5 trial types
    Fig S5C - raster grid, 3 additional target regions (TS, NAc_c, OT) x 5 trial types
    Fig S5D - GRAB-DA amplitude by trial type
              NOTE: this plots all 7 regions overlaid, but the Fig S5D image
              shows only 3 lines (TS, NAc_c, OT) 

Data can be downloaded here: https://doi.org/10.17605/OSF.IO/DV724
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats


# %% Figure formatting

def set_up_figure_format():
    """Set consistent style/rcParams for all figures in this script."""
    sns.set_theme(
        font="Helvetica", font_scale=0.75, style='ticks',
        rc={"axes.spines.right": False, "axes.spines.top": False},
        palette=["#ff595e", "#6a4c93", "#ff924c", "#ffca3a", "#8ac926", "#52a675", "#4267ac"],
    )

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    matplotlib.rcParams['axes.linewidth'] = 0.5
    matplotlib.rcParams['ytick.major.width'] = 0.5
    matplotlib.rcParams['xtick.major.width'] = 0.5
    matplotlib.rcParams['xtick.minor.width'] = 0.5
    matplotlib.rcParams['ytick.minor.width'] = 0.5
    matplotlib.rcParams['xtick.major.size'] = 3
    matplotlib.rcParams['ytick.major.size'] = 3
    matplotlib.rcParams['xtick.minor.size'] = 1.5
    matplotlib.rcParams['ytick.minor.size'] = 1.5
    matplotlib.rcParams['lines.linewidth'] = 0.5

    plt.close('all')


# %% Plotting helpers

def plot_raster(data, x=None, ax=None, vmin=None, vmax=None):
    """Plot a trial (or session-average) x time heatmap on the given axis."""
    if x is None:
        x = np.arange(data.shape[1])
    if ax is None:
        ax = plt.gca()

    yaxis = np.arange(len(data))
    ax.pcolormesh(x, yaxis, data, cmap=sns.color_palette('gray_r', as_cmap=True), vmin=vmin, vmax=vmax)
    ax.set_yticks([0, len(data) - 1])


def bounded_plot(y, x=None, ax=None, axis=0, label='', color='b'):
    """Plot mean +/- SEM of `y` along `axis`."""
    if x is None:
        x = np.arange(y.T.shape[axis])
    if ax is None:
        _, ax = plt.subplots()

    m = np.nanmean(y, axis=axis)
    sem = stats.sem(y, axis=axis, nan_policy='omit')
    ax.fill_between(x, m + sem, m - sem, alpha=0.3, color=color, lw=0)
    ax.plot(x, m, label=label, color=color)


# %% Load data and select sessions
#
# Two-stage selection, additive (OR'd into the same mask, not reset in
# between): stage 1 keeps multi-site sessions across the 4 main regions;
# stage 2 adds sessions recorded from the 3 additional-target regions. The
# combined set is what feeds both the S2C and S5C raster panels below.

set_up_figure_format()

DATA_PATH = 'Data/av_and_probcond_raster.npy'
data = np.load(DATA_PATH, allow_pickle=True).item()
trmtx = data['trmtx']
raster_all_data = data['raster']
t_raster = data['t_raster']
ls_sess = trmtx['sessid'].unique()

idx_select = np.zeros((len(trmtx)), dtype=bool)

# Stage 1: multi-site recordings across the 4 main regions (+ NAc_c, kept
# distinct here rather than merged into NAc_lat, since NAc_c is its own
# region of interest for Fig S5).
ls_loc_multisite_filter = ['DS', 'NAc_lat', 'BLA', 'mPFC', 'NAc_c']
for sess in ls_sess:
    sub_loc = trmtx['loc'][trmtx['sessid'] == sess].unique()
    exp = trmtx['experiment'][trmtx['sessid'] == sess].unique()
    if len(sub_loc) > 2:
        if set(sub_loc).issubset(ls_loc_multisite_filter):
            if set(['Av']).issubset(exp):
                idx_select = idx_select | ((trmtx['sessid'] == sess).to_numpy() & (trmtx['experiment'] == 'Av').to_numpy())

# Stage 2 (additive): sessions recorded from the additional target regions
ls_loc_target_filter = ['TS', 'NAc_c', 'OT']
for sess in ls_sess:
    sub_loc = trmtx['loc'][trmtx['sessid'] == sess].unique()
    exp = trmtx['experiment'][trmtx['sessid'] == sess].unique()
    if len(sub_loc) > 0:
        if set(sub_loc).issubset(ls_loc_target_filter):
            if set(['Av']).issubset(exp):
                idx_select = idx_select | ((trmtx['sessid'] == sess).to_numpy() & (trmtx['experiment'] == 'Av').to_numpy())

trmtx_sub = trmtx.iloc[idx_select, :].copy()
raster_sub = raster_all_data[idx_select, :]

# Crop raster and adjust timing
t_raster += 0.125
crop_win = [-0.5, 3.5]
raster_sub = raster_sub[:, (t_raster > crop_win[0]) & (t_raster < crop_win[1])]
t_raster = t_raster[(t_raster > crop_win[0]) & (t_raster < crop_win[1])]

ls_loc_all = ['DS', 'TS', 'NAc_lat', 'NAc_c', 'OT', 'BLA', 'mPFC']


# %% Label trial types

ls_sess = trmtx_sub['sessid'].unique()

ls_color_trial = ['dodgerblue', 'lightseagreen', 'grey', 'crimson', 'slateblue']
ls_trial = [[3, 5], [2, 5], [2, 0], [3, -1], [2, -1]]
ls_trial_type = ['uncued rew', 'cued rew', 'omission', 'uncued pun', 'omission pun']
ls_ylabel = ['Uncued\nreward', 'Cued\nreward', 'Omission', 'Air puff', 'Omission &\n air puff']

tr_type = np.zeros(len(trmtx_sub), dtype=object)
for j, trial_selector in enumerate(ls_trial):
    tr_type[(trmtx_sub['ToneID'] == trial_selector[0]) & (trmtx_sub['Reward?'] == trial_selector[1])] = ls_trial_type[j]
tr_type[tr_type == 0] = 'NA'
trmtx_sub['trial_type'] = tr_type
trmtx_sub['trial_type'] = pd.Categorical(trmtx_sub['trial_type'], categories=ls_trial_type, ordered=True)
trmtx_sub['loc'] = pd.Categorical(trmtx_sub['loc'], categories=ls_loc_all, ordered=True)

trmtx_sub['grab_cue'] = np.mean(raster_sub[:, (t_raster > 0) & (t_raster < 1)], axis=1)
trmtx_sub['grab_reinf'] = np.mean(raster_sub[:, (t_raster > 1.5) & (t_raster < 2.5)], axis=1)

mean_sess = trmtx_sub.groupby(['sessid', 'loc', 'trial_type'])[['grab_cue', 'grab_reinf']].mean()


# %% Session-averaged raster data (shared by the S2C and S5C panels below)

gidx = trmtx_sub.groupby(['sessid', 'loc', 'trial_type'], observed=True).indices

labels = []
means = []
for key, row_idx in gidx.items():
    labels.append(key)
    means.append(np.nanmean(raster_sub[np.fromiter(row_idx, dtype=int)], axis=0))

ras_sess_averages = np.stack(means)  # shape: (n_groups, n_timepoints)
group_labels = pd.DataFrame(labels, columns=['an', 'loc', 'trial_type'])


# %% Fig S2C: raster grid, 4 main regions

ls_ylabel = ['Uncued reward', 'Cued reward', 'Omission', 'Air puff', 'Omission & air puff']
ls_col_loc_main4 = ["#ff595e", "#ff924c", "#52a675", "#4267ac"]
ls_loc_main4 = ['DS', 'NAc_lat', 'BLA', 'mPFC']

fig, ax = plt.subplots(5, 5, figsize=(8, 6), sharex=True)
for i, loc in enumerate(ls_loc_main4):
    for j, trial in enumerate(ls_trial_type):
        idx_select = (group_labels['loc'] == loc) & (group_labels['trial_type'] == trial)

        # Sort order (by peak response) computed once from 'uncued rew',
        # reused for the other trial types in this row.
        if j == 0:
            X = ras_sess_averages[idx_select]
            m = np.mean(X[:, (t_raster > 1.5) & (t_raster < 2.5)], axis=1)
            idx_sort = np.argsort(m)

        a = ax[i, j]
        plot_raster(ras_sess_averages[idx_select][idx_sort], x=t_raster, ax=a, vmin=-1.5, vmax=5)

        if j == 0:
            a.set_ylabel(loc, color=ls_col_loc_main4[i])
        if i == 0:
            a.set_title(ls_ylabel[j])

        if j == 0 or j == 3:
            a.axvline(1.5, ls='-', color='k', alpha=0.5)
        elif j == 1 or j == 4:
            a.axvline(0, ls='-', color='k', alpha=0.5)
            a.axvline(1.5, ls='-', color='k', alpha=0.5)
        else:
            a.axvline(0, ls='-', color='k', alpha=0.5)

        a = ax[4, j]
        bounded_plot(ras_sess_averages[idx_select], x=t_raster, ax=a, color=ls_col_loc_main4[i])
        if i == 0 and j == 0:
            a.set_ylabel('GRAB-DA\n(z-score)')
            a.set_xlabel('Time from cue (s)')
        a.set_ylim(-2.5, 7)
fig.tight_layout()


# %% Fig S5C: raster grid, 3 additional target regions

ls_col_loc_targets3 = ["#6a4c93", "#ffca3a", "#8ac926"]
ls_loc_targets3 = ['TS', 'NAc_c', 'OT']

fig, ax = plt.subplots(4, 5, figsize=(8, 5), sharex=True)
for i, loc in enumerate(ls_loc_targets3):
    for j, trial in enumerate(ls_trial_type):
        idx_select = (group_labels['loc'] == loc) & (group_labels['trial_type'] == trial)

        if j == 0:
            X = ras_sess_averages[idx_select]
            m = np.mean(X[:, (t_raster > 1.5) & (t_raster < 2.5)], axis=1)
            idx_sort = np.argsort(m)

        a = ax[i, j]
        plot_raster(ras_sess_averages[idx_select][idx_sort], x=t_raster, ax=a, vmin=-1.5, vmax=5)

        if j == 0:
            a.set_ylabel(loc, color=ls_col_loc_targets3[i])
        if i == 0:
            a.set_title(ls_ylabel[j])

        if j == 0 or j == 3:
            a.axvline(1.5, ls='-', color='k', alpha=0.5)
        elif j == 1 or j == 4:
            a.axvline(0, ls='-', color='k', alpha=0.5)
            a.axvline(1.5, ls='-', color='k', alpha=0.5)
        else:
            a.axvline(0, ls='-', color='k', alpha=0.5)

        a = ax[3, j]
        bounded_plot(ras_sess_averages[idx_select], x=t_raster, ax=a, color=ls_col_loc_targets3[i])
        if i == 0 and j == 0:
            a.set_ylabel('GRAB-DA\n(z-score)')
            a.set_xlabel('Time from cue (s)')
        a.set_ylim(-2.5, 7)
fig.tight_layout()


# %% Fig S5D: GRAB-DA amplitude by trial type
# See note at top of file re: this overlaying all 7 regions vs. the 3 shown
# in the published panel.

ls_xticks = ['Uncued rew.', 'Cued reward', 'Omission', 'Air puff', 'Omission & air']
fig, ax = plt.subplots(figsize=(3, 3))
a = ax
sns.lineplot(mean_sess, x='trial_type', y='grab_reinf', hue='loc',
             err_style='bars', marker='o', lw=1,
             palette=["#ff595e", "#6a4c93", "#ff924c", "#ffca3a", "#8ac926", "#52a675", "#4267ac"],
             ax=a)
a.axhline(0, ls='--', color='k', alpha=0.5)
a.set_xlabel('')
a.set_ylabel('GRAB-DA amplitude (z-score)')
a.set_xticklabels(ls_xticks, rotation=90)
a.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
fig.tight_layout()
