#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 21:07:57 2025

@author: vbp

Trial-history effect of outcome on anticipatory licking, for Figure 2B and
Figure S2D.

    Fig S2D - delta anticipatory lick rate relative to baseline, across
              trials before/after cued reward, omission, and omission+air
              puff, vs. a trial-shuffled control (ax[0] below)
    Fig 2B  - that same effect isolated at the trial immediately after the
              outcome, for cued trials (ax[1]) and uncued trials (ax[2])

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


def makeRaster(a, x, w):
    """
    Build a raster of `a` aligned to each index in `x`, spanning offsets
    w[0] (inclusive) to w[1] (exclusive) around each index. Out-of-bounds
    windows (near the start/end of `a`) are padded with NaN rather than
    wrapping or erroring.

    Parameters
    ----------
    a : 1D array to build the raster from
    x : integer indices to align to
    w : [start_offset, end_offset], e.g. [-10, 100]

    Returns
    -------
    R : raster, shape (len(x), w[1] - w[0])
    """
    if x[-1] + w[1] > a.shape[0]:
        nMissing = int(x[-1] + w[1] - a.shape[0] + 1)
        a = np.append(a, np.ones(nMissing) * np.nan)

    if x[0] + w[0] < 0:
        nMissing = int(-x[0] - w[0])
        a = np.concatenate((np.ones(nMissing) * np.nan, a))
        x = x.copy() + nMissing

    X = np.tile(np.arange(np.diff(w)) + w[0], (len(x), 1))
    Y = X.transpose() + x
    Y = Y.transpose()
    Y = Y.flatten()
    Y = Y.astype(int)

    R = a[Y]
    R = R.reshape((X.shape[0], X.shape[1]))

    return R


# %% Load data and select recordings

set_up_figure_format()

DATA_PATH = 'Data/av_and_probcond_raster.npy'
data = np.load(DATA_PATH, allow_pickle=True).item()
trmtx = data['trmtx']
ls_sess = trmtx['sessid'].unique()

ls_loc = ['DS', 'TS', 'NAc_lat', 'NAc_c', 'OT', 'BLA', 'mPFC']

idx_select = np.zeros((len(trmtx)), dtype=bool)
for sess in ls_sess:
    sub_loc = trmtx['loc'][trmtx['sessid'] == sess].unique()
    exp = trmtx['experiment'][trmtx['sessid'] == sess].unique()
    if len(sub_loc) > 0:
        if set(sub_loc).issubset(ls_loc):
            if set(['Av']).issubset(exp):
                idx_select = idx_select | ((trmtx['sessid'] == sess).to_numpy() & (trmtx['experiment'] == 'Av').to_numpy())
trmtx_sub = trmtx.iloc[idx_select, :].copy()


# %% Label trial types

ls_sess = trmtx_sub['sessid'].unique()

ls_trial = [[3, 5], [2, 5], [2, 0], [3, -1], [2, -1]]
ls_trial_type = ['uncued rew', 'cued rew', 'omission', 'uncued pun', 'omission pun']

tr_type = np.zeros(len(trmtx_sub), dtype=object)
for j, trial_selector in enumerate(ls_trial):
    tr_type[(trmtx_sub['ToneID'] == trial_selector[0]) & (trmtx_sub['Reward?'] == trial_selector[1])] = ls_trial_type[j]
tr_type[tr_type == 0] = 'NA'
trmtx_sub['trial_type'] = tr_type
trmtx_sub['trial_type'] = pd.Categorical(trmtx_sub['trial_type'], categories=ls_trial_type, ordered=True)
trmtx_sub['loc'] = pd.Categorical(trmtx_sub['loc'], categories=ls_loc, ordered=True)


# %% Trial-history effect: anticipatory lick rate before/after each outcome

rng = np.random.default_rng(25)
results = []
win = [-2, 5]  # trial offsets relative to the outcome: -2 .. +4

ls_history_trial_types = ['cued rew', 'omission', 'omission pun', 'uncued rew', 'uncued pun']

for sess in ls_sess:
    sess_data = trmtx_sub[trmtx_sub['sessid'] == sess][
        ['Trial#', 'trial_type', 'timeTone', 'BL(-1) l/s', 'Anticipatory(tUS-CS) l/s', 'Consumatory(+2) l/s']
    ].copy()
    sess_data = sess_data.rename(columns={
        'Trial#': 'trial_id', 'timeTone': 't_tone', 'BL(-1) l/s': 'bl',
        'Anticipatory(tUS-CS) l/s': 'anticip', 'Consumatory(+2) l/s': 'conso',
    })
    sess_data = sess_data.reset_index(drop=True)

    # Anticipatory licking is only meaningful on cued trials (there's no
    # cue to anticipate on uncued trials), so blank it elsewhere. The
    # history windows below can still land on cued trials near an uncued
    # outcome and pick up a real value there.
    sess_data['iscue'] = (sess_data['trial_type'] == 'cued rew') | (sess_data['trial_type'] == 'omission') | (sess_data['trial_type'] == 'omission pun')

    anti = sess_data['anticip'].to_numpy()
    anti[~sess_data['iscue'].to_numpy()] = np.nan
    avg_anticip = np.nanmean(anti)

    # Exclude sessions with weak/unreliable anticipatory licking overall
    # (avg cued-trial anticipatory rate <= 1 lick/s)
    if not (avg_anticip > 1):
        continue

    for trial in ls_history_trial_types:
        idx = np.where(sess_data['trial_type'] == trial)[0]
        if len(idx) == 0:
            continue

        history = makeRaster(anti, idx, win)
        m_history = np.nanmean(history, axis=0)
        m_history -= np.nanmean(m_history[:-win[0]])  # baseline: pre-outcome trials

        # Trial-shuffled control (destroys any true trial-history structure)
        anti_sh = anti.copy()
        rng.shuffle(anti_sh)
        history_sh = makeRaster(anti_sh, idx, win)
        m_history_sh = np.nanmean(history_sh, axis=0)
        m_history_sh -= np.nanmean(m_history_sh[:-win[0]])

        for w, m, m_sh in zip(np.arange(win[1] - win[0]) + win[0], m_history, m_history_sh):
            results.append({
                'sess': sess,
                'trial_type': trial,
                'trial #': w,
                'anticip': m,
                'anticip_sh': m_sh,
            })

results = pd.DataFrame(results)
# Explicit ordering so hue colors below map deterministically to trial
# type, rather than relying on first-appearance order in the data.
results['trial_type'] = pd.Categorical(results['trial_type'], categories=ls_history_trial_types, ordered=True)


# %% Fig S2D (ax[0]) and Fig 2B (ax[1], ax[2])

ls_col_cue = ['lightseagreen', 'gray', 'slateblue']  # cued rew, omission, omission pun

fig, ax = plt.subplots(1, 3, figsize=(5, 3), sharey=True)

idx_cue = (results['trial_type'] == 'cued rew') | (results['trial_type'] == 'omission') | (results['trial_type'] == 'omission pun')

# Fig S2D: full trial-history trajectory, cued trial types
a = ax[0]
sns.lineplot(results[idx_cue], x='trial #', y='anticip_sh', ax=a,
             errorbar='se', err_style='bars', marker='o', color='w', ls='--',
             markeredgecolor='k', err_kws={'ecolor': 'k', 'linewidth': 0.5})
sns.lineplot(results[idx_cue], x='trial #', y='anticip', hue='trial_type', ax=a,
             errorbar='se', err_style='bars', marker='o', palette=ls_col_cue)
a.set_ylabel('Δ anticipatory lickrate (l/s)')
a.set_xticks(np.arange(win[1] - win[0]) + win[0])
a.axvline(0.5, lw=0.5, ls=':', color='k', alpha=0.5)

# Fig 2B: effect isolated at trial +1, cued trial types
a = ax[1]
idx_t1_cue = idx_cue & (results['trial #'] == 1)
sns.lineplot(results[idx_t1_cue], x='trial_type', y='anticip', hue='trial_type',
             ax=a, palette=ls_col_cue, errorbar='se', err_style='bars', marker='o', legend=False)
sns.lineplot(results[idx_t1_cue], x='trial_type', y='anticip_sh', hue='trial_type',
             ax=a, palette=['w', 'w', 'w'], errorbar='se', err_style='bars', marker='o', legend=False,
             markeredgecolor='k', err_kws={'ecolor': 'k', 'linewidth': 0.5})
a.axhline(0, lw=0.5, ls=':', color='k', alpha=0.5)
a.set_ylabel('Δ anticipatory lickrate to CS+ (l/s)')
a.set_xlim(-0.5, 2.5)
a.set_xlabel('')
plt.setp(a.get_xticklabels(), rotation=90, ha='right')

# Fig 2B: effect isolated at trial +1, uncued trial types
a = ax[2]
idx_uncued_t1 = ((results['trial_type'] == 'uncued rew') | (results['trial_type'] == 'uncued pun')) & (results['trial #'] == 1)
sns.lineplot(results[idx_uncued_t1], x='trial_type', y='anticip', hue='trial_type',
             ax=a, palette=['dodgerblue', 'crimson'], errorbar='se', err_style='bars', marker='o', legend=False)
a.axhline(0, lw=0.5, ls=':', color='k', alpha=0.5)
a.set_ylabel('Δ anticipatory lickrate to CS+ (l/s)')
a.set_xlim(-0.5, 1.5)
a.set_xlabel('')
plt.setp(a.get_xticklabels(), rotation=90, ha='right')

fig.tight_layout()


# %% Stats: post-outcome anticipatory lick rate vs. shuffled control
#
# `anticip` and `anticip_sh` are paired within each (session, trial_type,
# trial #) row — same session, real vs. shuffled — so this is a paired
# comparison. Using ttest_rel rather than ttest_ind.

for lag in [1, 3]:
    print(f'\n=== trial # {lag} ===')
    for trial in ['cued rew', 'omission', 'omission pun']:
        idx = (results['trial_type'] == trial) & (results['trial #'] == lag)
        real = results['anticip'][idx].to_numpy()
        shuf = results['anticip_sh'][idx].to_numpy()
        print(f'post-{trial} vs shuffled')
        print(stats.ttest_rel(real, shuf, nan_policy='omit'))

# NOTE: this next comparison (omission vs. omission+punishment) may also be
# paired if the same sessions contribute both trial types — worth checking
# whether ttest_rel is more appropriate here too, rather than ttest_ind.
print('\npost-omission vs post-omission+pun')
print(stats.ttest_ind(
    results['anticip'][(results['trial_type'] == 'omission') & (results['trial #'] == 1)],
    results['anticip'][(results['trial_type'] == 'omission pun') & (results['trial #'] == 1)],
    nan_policy='omit',
))
