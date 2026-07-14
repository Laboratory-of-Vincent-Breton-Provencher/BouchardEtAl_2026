#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 15:51:12 2025

@author: vbp

Time-resolved regression weights for Figure 3I and 3J.

For each session x region, refits the R/S regression model (using each
session's already-fitted alpha, delta from the non-cross-validated
regression script) independently at every time point around
reinforcement, producing a beta_motiv (R) and beta_senso (S) trace over
time.

    Fig 3I - beta_motiv and beta_senso weight traces over time, one panel
              per region (the final 1x4 figure below)
    Fig 3J - lag between the sensory and motivational weight traces,
              extracted via cross-correlation, by region and combined

Requires `results_multisite_regression2.csv` (best_value) already computed
by the non-cross-validated regression script: Figure_3B-H_S4ACDEFH.py

Data can be downloaded here: https://doi.org/10.17605/OSF.IO/DV724
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge


# %% Figure formatting

def set_up_figure_format():
    """Set consistent style/rcParams for all figures in this script."""
    sns.set_theme(
        font="Helvetica", font_scale=0.75, style='ticks',
        rc={"axes.spines.right": False, "axes.spines.top": False},
        palette=["#ff595e", "#8ac926", "#1982c4", "#6a4c93", "#ff924c", "#ffca3a", "#52a675", "#4267ac", "#6a4c93"],
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
    matplotlib.rcParams['image.cmap'] = 'mako'
    matplotlib.rcParams['lines.linewidth'] = 0.75

    plt.close('all')


# %% Model encodings

def encode_motivational(trmtx, alpha, negative_scale=1, punish_value=-1.0):
    """Encode each trial's motivational (reward-value) regressor R."""
    raw = trmtx[['ToneID', 'Reward?']].to_numpy()
    out = []
    for v in raw:
        if v[0] == 3 and v[1] == 5:      # Uncued reward
            out.append(1)
        elif v[0] == 2 and v[1] == 5:    # Cued reward
            out.append(1 - alpha)
        elif v[0] == 2 and v[1] == 0:    # Reward omission
            out.append((-alpha) * negative_scale)
        elif v[0] == 3 and v[1] < 0:     # Uncued air puff
            out.append(punish_value * negative_scale)
        elif v[0] == 2 and v[1] < 0:     # Omission + air puff
            out.append((-alpha + punish_value) * negative_scale)
        elif v[0] == 3 and v[1] == 0:    # Nothing
            out.append(0)
        else:
            raise ValueError(f"Unexpected motivational level (Tone='{v[0]}', Outcome='{v[1]}')")
    out = np.array(out)
    return out / np.std(out)


def encode_sensory(trmtx, delta):
    """Encode each trial's sensory-intensity regressor S."""
    raw = trmtx[['ToneID', 'Reward?']].to_numpy()
    out = []
    for v in raw:
        if v[0] == 3 and v[1] == 5:      # Uncued reward
            out.append(1 - delta)
        elif v[0] == 2 and v[1] == 5:    # Cued reward
            out.append(1 - delta)
        elif v[0] == 2 and v[1] == 0:    # Reward omission
            out.append(0)
        elif v[0] == 3 and v[1] < 0:     # Uncued air puff
            out.append(1)
        elif v[0] == 2 and v[1] < 0:     # Omission + air puff
            out.append(1)
        elif v[0] == 3 and v[1] == 0:    # Nothing
            out.append(0)
        else:
            raise ValueError(f"Unexpected motivational level (Tone='{v[0]}', Outcome='{v[1]}')")
    out = np.array(out)
    return out / np.std(out)


# %% Load data and select multi-site recordings (PFC, NAc_lat, DS, BLA)

set_up_figure_format()

DATA_PATH = 'Data/av_and_probcond_raster.npy'
data = np.load(DATA_PATH, allow_pickle=True).item()
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

t_raster += 0.125
crop_win = [-1, 7.5]
raster_sub = raster_sub[:, (t_raster > crop_win[0]) & (t_raster < crop_win[1])]
t_raster = t_raster[(t_raster > crop_win[0]) & (t_raster < crop_win[1])]

# Some fixing for DA4fib01
trmtx_sub['loc'] = trmtx_sub['loc'].replace({'NAc_c': 'NAc_lat'})
ls_loc = ['DS', 'NAc_lat', 'BLA', 'mPFC']
ls_sess = trmtx_sub['sessid'].unique()


# %% Model parameters and baseline subtraction

ls_trial = [
    [3, 5],   # Uncued reward
    [2, 5],   # Cued reward
    [2, 0],   # Reward omission
    [3, -1],  # Uncued air puff
    [2, -1],  # Reward omission & air puff
]
ls_trial_names = ['uncued rew', 'cued rew', 'omission', 'uncued pun', 'omission+pun']

fit_win = [1.25, 3.5]  # window used to fit the model (relative to cue)
negative_scale = 0.5

bl = np.mean(raster_sub[:, (t_raster > fit_win[0] - 0.25) & (t_raster < fit_win[0])], axis=1)
raster_sub = (raster_sub.T - bl).T

raster_sub_crop = raster_sub[:, (t_raster > fit_win[0]) & (t_raster < fit_win[1])]
t_beta = t_raster[(t_raster > fit_win[0]) & (t_raster < fit_win[1])]
t_beta -= 1.5  # re-center on reinforcement

best_value = pd.read_csv('Data/results_multisite_regression2.csv')


# %% Fit the R/S model independently at every time point, per session x region

best_alpha_by_sess = best_value.groupby('sess')['alpha_learning'].mean()

results = []
for sess in ls_sess:
    alpha = best_alpha_by_sess[sess]
    motivational = encode_motivational(trmtx_sub, alpha, negative_scale=negative_scale)

    for loc in ls_loc:
        idx = (best_value['loc'] == loc) & (best_value['sess'] == sess)
        if idx.sum() == 0:
            continue
        delta = best_value['delta_punish'].loc[idx].to_numpy()[0]
        sensory = encode_sensory(trmtx_sub, delta)

        print(f'{sess} {loc}: alpha={alpha} delta={delta}')

        idx_trial_select = ((trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == loc)).to_numpy()
        X = np.array([motivational[idx_trial_select], sensory[idx_trial_select]]).T

        for ti, t in enumerate(t_beta):
            y = raster_sub_crop[idx_trial_select, ti]

            clf = Ridge(alpha=0)
            clf.fit(X, y)

            results.append({
                'sess': sess,
                'loc': loc,
                'alpha': alpha,
                'delta': delta,
                't': t,
                'b_motiv': clf.coef_[0],
                'b_senso': clf.coef_[1],
                'senso_bias': (np.abs(clf.coef_[1]) - np.abs(clf.coef_[0])) / (np.abs(clf.coef_[0]) + np.abs(clf.coef_[1])),
                'r2_score': clf.score(X, y),
            })

results = pd.DataFrame(results)


# %% Peak-normalize beta_motiv/beta_senso per session x region

results['b_motiv_peak'] = 0.0
results['b_senso_peak'] = 0.0

for sess in ls_sess:
    for loc in ls_loc:
        idx_select = (results['sess'] == sess) & (results['loc'] == loc)
        if idx_select.sum() > 0:
            results.loc[idx_select, 'b_motiv_peak'] = results.loc[idx_select, 'b_motiv'] / np.max(results.loc[idx_select, 'b_motiv'])
            results.loc[idx_select, 'b_senso_peak'] = results.loc[idx_select, 'b_senso'] / np.max(results.loc[idx_select, 'b_senso'])


# %% Quick diagnostic: all regions overlaid (not a numbered figure panel)

fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.5))
for i, y in enumerate(['b_motiv', 'b_senso', 'senso_bias']):
    a = ax[i]
    sns.lineplot(results, x='t', y=y, hue='loc', ax=a)
fig.tight_layout()


# %% Fig 3I: beta_motiv (R) and beta_senso (S) weight traces by region

fig, ax = plt.subplots(1, 4, figsize=(7, 2.5), sharex=True, sharey=True)
for i, loc in enumerate(ls_loc):
    sub = results[results['loc'] == loc]

    a = ax[i]
    sns.lineplot(sub, x='t', y='b_motiv', ax=a, color='k', errorbar='se', label='b_motiv')
    sns.lineplot(sub, x='t', y='b_senso', ax=a, color='crimson', errorbar='se', label='b_senso')

    a.set_title(loc)
    a.set_ylabel('Weights')
    a.axhline(0, ls=':', alpha=0.5, color='k')

    if i != 0:
        a.legend().remove()
        a.set_xlabel('')
    else:
        a.set_xlabel('Time from reinforcement (s)')
fig.tight_layout()


# %% Diagnostic: raw cross-correlation traces (supports Fig 3J below)

results_lag = []
fig, ax = plt.subplots(figsize=(2.5, 2.6))
for sess in ls_sess:
    for loc in ls_loc:
        idx_trial_select = (results['sess'] == sess) & (results['loc'] == loc)
        if idx_trial_select.sum() == 0:
            continue

        bm_win = results[idx_trial_select]['b_motiv'].to_numpy()[(t_beta > 0) & (t_beta < 1)]
        bs_win = results[idx_trial_select]['b_senso'].to_numpy()[(t_beta > 0) & (t_beta < 1)]

        xcorr = np.correlate(bs_win, bm_win, mode='full')
        t_corr = (np.arange(xcorr.shape[0]) - (xcorr.shape[0] - 1) / 2) * np.mean(np.diff(t_beta))

        ax.plot(t_corr, xcorr)
        lag = t_corr[np.argmax(xcorr)]
        print(f'{sess} {loc} lag={lag:0.3f}')

        results_lag.append({'sess': sess, 'loc': loc, 'lag': lag})

ax.axvline(0, ls='--', color='k')
fig.tight_layout()

results_lag = pd.DataFrame(results_lag)


# %% Fig 3J: lag between sensory and motivational weight traces, by region and combined

fig, ax = plt.subplots(1, 2, figsize=(3, 2.5), sharey=True)

a = ax[0]
sns.lineplot(results_lag, x='loc', y='lag', errorbar='se', color='k', err_style='bars', marker='o', ax=a)
a.axhline(0, ls='--', color='k')
a.set_ylabel('Lag sensory-motivational (s)')
a.set_xlabel('')
plt.setp(a.get_xticklabels(), rotation=90, ha='center')

a = ax[1]
a.errorbar(0, results_lag['lag'].mean(), results_lag['lag'].sem(), marker='o', color='k',
           markersize=6, markeredgecolor='w', markeredgewidth=0.75)
a.axhline(0, ls='--', color='k')
a.set_xticks([0])
a.set_ylabel('Lag sensory-motivational (s)')
a.set_xlabel('All')
_, p_all = stats.ttest_1samp(results_lag['lag'], 0)
a.set_title('p={:0.3}'.format(p_all))

fig.tight_layout()
