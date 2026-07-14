#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 13:26:45 2025

@author: vbp

Trial-to-trial ("noise") correlation analysis for Figure 5 and Figure S6.

    Fig 5A  - baseline noise-correlation scatter plots, one example session
    Fig 5B  - baseline noise correlation vs. shuffled control, by region pair
    Fig 5C  - time-resolved noise correlation, uncued reward / uncued punishment
    Fig 5D  - time-resolved noise correlation, combined cued trials ("CS+")
    Fig S6A - time-resolved noise correlation, cued reward / omission / omission+air
              (produced by the SAME grid as Fig 5C — just the other 3 of 5 rows)
    Fig S6B - time-resolved noise correlation across reward sizes (0.3-10 uL)

Fig 5C and S6A come from one 5-row grid (all 5 trial types); the published
figures split that single grid's rows across two panels.

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
        palette=["#ff595e", "#ff924c", "#52a675", "#4267ac"],
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


def compute_noise_corr_timeresolved(trmtx_sub, raster_sub_bin, ls_sess, ls_loc, trial,
                                     trial_col, value_col, ref_idx, rng_seed):
    """
    For one trial type, compute the time-resolved pairwise noise
    correlation across `ls_loc` (lower-triangle region pairs, in the
    order given by `np.tril(np.ones((n_loc, n_loc)), -1) > 0`), a
    trial-shuffled control, and baseline-correct each session's trace by
    subtracting its value at time-bin index `ref_idx`.

    Returns (noise_corr_time, noise_corr_sh_time), each shaped
    (n_sessions, n_timepoints, n_pairs).
    """
    rng = np.random.default_rng(rng_seed)
    n_loc = len(ls_loc)
    tril_mask = np.tril(np.ones((n_loc, n_loc)), -1) > 0

    noise_corr_time = []
    noise_corr_sh_time = []
    for sess in ls_sess:
        idx_select = (trmtx_sub['sessid'] == sess) & (trmtx_sub[trial_col] == trial)
        x = trmtx_sub.loc[idx_select][['loc', trial_col, value_col]]

        n_trial = 0
        for loc in ls_loc:
            if np.sum(x['loc'] == loc) > 0:
                n_trial = np.sum(x['loc'] == loc)

        shuffle_idx = np.argsort(rng.random((n_trial, n_loc)), axis=0)

        noise_corr = []
        noise_corr_sh = []
        for t in range(raster_sub_bin.shape[1]):
            df_corr = []
            for loc in ls_loc:
                if np.sum(x['loc'] == loc) > 0:
                    df_corr.append(raster_sub_bin[(idx_select & (trmtx_sub['loc'] == loc)).to_numpy(), t])
                else:
                    df_corr.append(np.full(n_trial, np.nan))

            noise_corr.append(pd.DataFrame(np.array(df_corr).T, columns=ls_loc).corr().to_numpy())
            shuffled = np.take_along_axis(np.array(df_corr).T, shuffle_idx, axis=0)
            noise_corr_sh.append(pd.DataFrame(shuffled, columns=ls_loc).corr().to_numpy())

        noise_corr = np.array(noise_corr)[:, tril_mask]
        noise_corr_sh = np.array(noise_corr_sh)[:, tril_mask]

        noise_corr = noise_corr - noise_corr[ref_idx, :]

        noise_corr_time.append(noise_corr)
        noise_corr_sh_time.append(noise_corr_sh)

    return np.array(noise_corr_time), np.array(noise_corr_sh_time)


def plot_noise_corr_grid(ax_row, noise_corr_time, t_raster_plot, color, ylim=None):
    """Plot one row of the noise-correlation grid (one trial type across
    region pairs), with a marker above each timepoint that's significantly
    different from zero (paired t-test, Bonferroni-corrected across the 8
    non-baseline timepoints)."""
    for j in range(noise_corr_time.shape[2]):
        a = ax_row[j]
        bounded_plot(noise_corr_time[:, :, j], t_raster_plot, ax=a, color=color)
        a.axvline(0, color='k', lw=0.5, ls=':', alpha=0.5)
        a.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)

        _, p = stats.ttest_rel(noise_corr_time[:, :, j], np.zeros_like(noise_corr_time[:, :, j]), nan_policy='omit')
        p_sig = np.array(p) * 8 < 0.05
        y_val = np.full(noise_corr_time.shape[1], 0.7)
        y_val[~p_sig] = np.nan
        a.plot(t_raster_plot, y_val, marker='o', markersize=3, color='k')
        if ylim is not None:
            a.set_ylim(*ylim)


# %% Load data and select multi-site recordings (PFC, NAc_lat, DS, BLA)

set_up_figure_format()

data = np.load('Data/av_and_probcond_raster.npy', allow_pickle=True).item()
trmtx = data['trmtx']
raster_all_data = data['raster']
t_raster = data['t_raster']
ls_sess = trmtx['sessid'].unique()

ls_loc = ['DS', 'NAc_lat', 'BLA', 'mPFC', 'NAc_c']
# ls_loc = ['DS','NAc_lat','BLA','mPFC','OT','NAc_c','NAc_m','TS']

idx_select = np.zeros((len(trmtx)), dtype=bool)
for sess in ls_sess:
    sub_loc = trmtx['loc'][trmtx['sessid'] == sess].unique()
    exp = trmtx['experiment'][trmtx['sessid'] == sess].unique()
    if len(sub_loc) > 2:
        if set(sub_loc).issubset(ls_loc):
            if set(['Av']).issubset(exp):
                idx_select = idx_select | ((trmtx['sessid'] == sess).to_numpy() & (trmtx['experiment'] == 'Av').to_numpy())
trmtx_sub = trmtx.iloc[idx_select, :].copy()
raster_sub = raster_all_data[idx_select, :]

# Crop raster and adjust timing (t=0 is cue onset here; reinforcement is
# 1.5s later, so time-resolved plots below re-center with a -1.5 offset)
t_raster += 0.125
crop_win = [-1, 3.5]
raster_sub = raster_sub[:, (t_raster > crop_win[0]) & (t_raster < crop_win[1])]
t_raster = t_raster[(t_raster > crop_win[0]) & (t_raster < crop_win[1])]

# Some fixing for DA4fib01
trmtx_sub['loc'] = trmtx_sub['loc'].replace({'NAc_c': 'NAc_lat'})
ls_loc = ['DS', 'NAc_lat', 'BLA', 'mPFC']
ls_sess = trmtx_sub['sessid'].unique()

ls_color_trial = ['dodgerblue', 'lightseagreen', 'grey', 'crimson', 'slateblue']
ls_trial = [[3, 5], [2, 5], [2, 0], [3, -1], [2, -1]]
ls_trial_type = ['uncued rew', 'cued rew', 'omission', 'uncued pun', 'omission pun']

tr_type = np.zeros(len(trmtx_sub), dtype=object)
for j, trial_selector in enumerate(ls_trial):
    tr_type[(trmtx_sub['ToneID'] == trial_selector[0]) & (trmtx_sub['Reward?'] == trial_selector[1])] = ls_trial_type[j]
tr_type[tr_type == 0] = 'NA'
trmtx_sub['trial_type'] = tr_type
trmtx_sub['trial_type'] = pd.Categorical(trmtx_sub['trial_type'], categories=ls_trial_type, ordered=True)
trmtx_sub['loc'] = pd.Categorical(trmtx_sub['loc'], categories=ls_loc, ordered=True)

trmtx_sub['grab_cue'] = np.mean(raster_sub[:, (t_raster > 0) & (t_raster < 1)], axis=1)
trmtx_sub['grab_reinf'] = np.mean(raster_sub[:, (t_raster > 1.5) & (t_raster < 2.5)], axis=1)
trmtx_sub['grab_bl'] = np.mean(raster_sub[:, (t_raster < 0)], axis=1)


# %% Fig 5A: baseline noise-correlation scatter plots (one example session)
# and Fig 5B: baseline noise correlation vs. shuffled control

ls_loc_corr = ['DS-NAc_lat', 'DS-BLA', 'NAc_lat-BLA', 'DS-mPFC', 'NAc_lat-mPFC', 'BLA-mPFC']
ls_loc_corr_ordered = ['DS-NAc_lat', 'BLA-mPFC', 'DS-mPFC', 'NAc_lat-mPFC', 'NAc_lat-BLA', 'DS-BLA']
ls_idx_ordered = [[0, 1], [2, 3], [1, 2], [1, 3], [0, 3], [0, 2]]

example_sess = 'DA4fib02_2024-02-26'  # for the Fig 5A scatter plots

rng = np.random.default_rng(25)
results_noise_corr = []

noise_corr_all = []
noise_corr_sh_all = []
for sess in ls_sess:
    idx_select = (trmtx_sub['sessid'] == sess)
    x = trmtx_sub.loc[idx_select][['loc', 'trial_type', 'grab_bl']]

    n_trial = 0
    for loc in ls_loc:
        if np.sum(x['loc'] == loc) > 0:
            n_trial = np.sum(x['loc'] == loc)

    df_corr = []
    for loc in ls_loc:
        if np.sum(x['loc'] == loc) > 0:
            df_corr.append(x[x['loc'] == loc]['grab_bl'].to_numpy())
        else:
            df_corr.append(np.full(n_trial, np.nan))

    noise_corr_all.append(pd.DataFrame(np.array(df_corr).T, columns=ls_loc).corr().to_numpy())

    shuffle_idx = np.argsort(rng.random(np.array(df_corr).T.shape), axis=0)
    shuffled = np.take_along_axis(np.array(df_corr).T, shuffle_idx, axis=0)
    noise_corr_sh_all.append(pd.DataFrame(shuffled, columns=ls_loc).corr().to_numpy())

    # Fig 5A: scatter plots for the one example session
    if sess == example_sess:
        fig, ax = plt.subplots(2, 3, figsize=(5.5, 3))
        ax = ax.flatten()
        for j, idx in enumerate(ls_idx_ordered):
            a = ax[j]
            a.scatter(df_corr[idx[0]], df_corr[idx[1]], s=3, c='k')
            if ~np.isnan(np.sum(df_corr[idx[0]] + df_corr[idx[1]])):
                r, p = stats.pearsonr(df_corr[idx[0]], df_corr[idx[1]])
                a.set_title(f'{ls_loc_corr_ordered[j]}\nr={r:0.2f} p={p:0.3f}', size=6)
            if j == 0:
                a.set_ylabel('GRAB (uncued reward)')
        fig.tight_layout()

noise_corr_all = np.array(noise_corr_all)
noise_corr_sh_all = np.array(noise_corr_sh_all)

tril_mask_4 = np.tril(np.ones((4, 4)), -1) > 0
noise_corr_all = noise_corr_all[:, tril_mask_4]
noise_corr_sh_all = noise_corr_sh_all[:, tril_mask_4]

for i in range(len(noise_corr_all)):
    for j, noise in enumerate(noise_corr_all[i]):
        results_noise_corr.append({'sess': ls_sess[i], 'shuffled': 'No', 'pair': ls_loc_corr[j], 'noise_corr': noise})
for i in range(len(noise_corr_sh_all)):
    for j, noise in enumerate(noise_corr_sh_all[i]):
        results_noise_corr.append({'sess': ls_sess[i], 'shuffled': 'Yes', 'pair': ls_loc_corr[j], 'noise_corr': noise})

results_noise_corr = pd.DataFrame(results_noise_corr)
results_noise_corr['pair'] = pd.Categorical(results_noise_corr['pair'], categories=ls_loc_corr_ordered, ordered=True)

fig, ax = plt.subplots(figsize=(3, 3))
a = ax
sns.boxplot(results_noise_corr, x='pair', y='noise_corr', hue='shuffled', ax=a,
            width=0.6, fliersize=1, palette=['gray', 'w'], linewidth=0.5)
plt.setp(a.get_xticklabels(), rotation=90, ha='right')
a.set_xlabel('')
a.set_ylabel('Trial to trial correlation')
a.legend()
fig.tight_layout()

for loc_corr in ls_loc_corr_ordered:
    x = results_noise_corr['noise_corr'][(results_noise_corr['pair'] == loc_corr) & (results_noise_corr['shuffled'] == 'No')]
    y = results_noise_corr['noise_corr'][(results_noise_corr['pair'] == loc_corr) & (results_noise_corr['shuffled'] == 'Yes')]
    p = stats.ttest_ind(x, y, nan_policy='omit')[1] * 6
    print(f'{loc_corr}: p = {p:0.1e}; n= {np.sum(~np.isnan(x))}')


# %% Fig 5C / S6A: time-resolved noise correlation, all 5 trial types
# (one 5x6 grid; the published figures split its rows: uncued
# reward/uncued punishment -> Fig 5C, the other 3 -> Fig S6A)

# Bin raster: 10-sample boxcar smoothing, then take every 5th sample
raster_sub_bin = np.array([np.convolve(x, np.ones(10) / 10, mode='same') for x in raster_sub])[:, 5::5]
t_raster_bin = t_raster[5::5]

# Baseline-correction reference bin index: uncued trials (no preceding
# cue) use bin 8; cued trials use bin 0 (true pre-cue baseline).
def ref_idx_probcond(trial):
    return 8 if trial in ('uncued rew', 'uncued pun') else 0


fig, ax = plt.subplots(5, 6, figsize=(8.5, 5), sharey=True, sharex=True)
for i, trial in enumerate(ls_trial_type):
    noise_corr_time, noise_corr_sh_time = compute_noise_corr_timeresolved(
        trmtx_sub, raster_sub_bin, ls_sess, ls_loc, trial,
        trial_col='trial_type', value_col='grab_reinf',
        ref_idx=ref_idx_probcond(trial), rng_seed=20,
    )
    # NOTE: noise_corr_sh_time (shuffled control) is computed here but not
    # currently used below — the plotted stats test each timepoint against
    # zero (ttest_rel), not against the shuffle. Kept in case you want to
    # reinstate the shuffle comparison; otherwise it's doubling this loop's
    # runtime for an unused result.
    plot_noise_corr_grid(ax[i], noise_corr_time, t_raster_bin - 1.5, ls_color_trial[i])
    for j in range(len(ls_loc_corr)):
        ax[i, j].set_xlim(-0.25, 2)
        ax[0, j].set_title(ls_loc_corr[j])
    ax[i, 0].set_ylabel(trial)
ax[4, 0].set_xlabel('Time from outcome (s)')
fig.tight_layout()


# %% Fig 5D: time-resolved noise correlation, combined cued trials ("CS+")

idx_cue = (trmtx_sub['trial_type'] == 'cued rew') | (trmtx_sub['trial_type'] == 'omission') | (trmtx_sub['trial_type'] == 'omission pun')
trmtx_cue = trmtx_sub.copy()
trmtx_cue['cue_group'] = np.where(idx_cue, 'CS+', 'NA')

noise_corr_time, noise_corr_sh_time = compute_noise_corr_timeresolved(
    trmtx_cue, raster_sub_bin, ls_sess, ls_loc, 'CS+',
    trial_col='cue_group', value_col='grab_reinf', ref_idx=0, rng_seed=20,
)

fig, ax = plt.subplots(1, 6, figsize=(8.5, 1.5), sharey=True)
plot_noise_corr_grid(ax, noise_corr_time, t_raster_bin + 0.2, 'k')
for j in range(len(ls_loc_corr)):
    ax[j].set_xlim(-0.5, 1.75)
    ax[j].set_title(ls_loc_corr[j])
ax[0].set_ylabel('CS+')
ax[0].set_xlabel('Time from cue (s)')
fig.tight_layout()


# %% Fig S6B: time-resolved noise correlation across reward sizes

data = np.load('Data/rewmag_raster.npy', allow_pickle=True).item()
trmtx = data['trmtx']
raster_all_data = data['raster']
t_raster = data['t_raster']
ls_sess = trmtx['sessid'].unique()

ls_loc = ['DS', 'NAc_lat', 'BLA', 'mPFC', 'NAc_c']

idx_select = np.zeros((len(trmtx)), dtype=bool)
for sess in ls_sess:
    sub_loc = trmtx['loc'][trmtx['sessid'] == sess].unique()
    if len(sub_loc) > 2:
        if set(sub_loc).issubset(ls_loc):
            idx_select = idx_select | ((trmtx['sessid'] == sess).to_numpy())
trmtx_sub = trmtx.iloc[idx_select, :].copy()
raster_sub = raster_all_data[idx_select, :]

t_raster += 0.125 - 1.5  # +1.5 to center on reinforcement
crop_win = [-1.5, 3.5]
raster_sub = raster_sub[:, (t_raster > crop_win[0]) & (t_raster < crop_win[1])]
t_raster = t_raster[(t_raster > crop_win[0]) & (t_raster < crop_win[1])]

trmtx_sub['loc'] = trmtx_sub['loc'].replace({'NAc_c': 'NAc_lat'})
ls_loc = ['DS', 'NAc_lat', 'BLA', 'mPFC']
ls_sess = trmtx_sub['sessid'].unique()

ls_color_trial_rew = sns.color_palette('mako_r', 5)
ls_trial = [[3, 0.3], [3, 1], [3, 2.5], [3, 5], [3, 10]]
ls_trial_type_rew = [0.3, 1, 2.5, 5, 10]

tr_type = np.zeros(len(trmtx_sub), dtype=object)
for j, trial_selector in enumerate(ls_trial):
    tr_type[(trmtx_sub['ToneID'] == trial_selector[0]) & (trmtx_sub['Reward?'] == trial_selector[1])] = ls_trial_type_rew[j]
tr_type[tr_type == 0] = np.nan
trmtx_sub['trial_type'] = tr_type
trmtx_sub['trial_type'] = pd.Categorical(trmtx_sub['trial_type'], categories=ls_trial_type_rew, ordered=True)
trmtx_sub['loc'] = pd.Categorical(trmtx_sub['loc'], categories=ls_loc, ordered=True)

trmtx_sub['grab_reinf'] = np.mean(raster_sub[:, (t_raster > 0) & (t_raster < 1)], axis=1)

raster_sub_bin = np.array([np.convolve(x, np.ones(10) / 10, mode='same') for x in raster_sub])[:, 5::5]
t_raster_bin = t_raster[5::5]

ref_idx_rewmag = 5

fig, ax = plt.subplots(5, 6, figsize=(8.5, 5), sharey=True, sharex=True)
for i, trial in enumerate(ls_trial_type_rew):
    noise_corr_time, noise_corr_sh_time = compute_noise_corr_timeresolved(
        trmtx_sub, raster_sub_bin, ls_sess, ls_loc, trial,
        trial_col='trial_type', value_col='grab_reinf',
        ref_idx=ref_idx_rewmag, rng_seed=20,
    )
    plot_noise_corr_grid(ax[i], noise_corr_time, t_raster_bin, ls_color_trial_rew[i])
    for j in range(len(ls_loc_corr)):
        ax[i, j].set_xlim(-0.25, 2)
        ax[0, j].set_title(ls_loc_corr[j])
    ax[i, 0].set_ylabel(trial)
ax[4, 0].set_xlabel('Time from outcome (s)')
fig.tight_layout()
