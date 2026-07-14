#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 05:36:17 2025

@author: vbp

Figure S2G only: correlation between a trial's post-outcome dopamine
response and the following trial's anticipatory licking, per region, vs.
a trial-shuffled control.

IMPORTANT: the original version of this analysis paired "this trial's
dopamine" with "next trial's licking" by shifting rows within a whole
session (all regions mixed together), rather than within a single
region's own trial sequence. Since the underlying table has one row per
(trial, location), that only pairs correctly-adjacent trials if rows
happen to already be sorted trial-then-... no, region-then-trial for each
session. This version instead explicitly groups by (session, region) and
sorts by trial number before shifting, which is correct regardless of the
original row order — but may change the resulting correlations if the
original ordering assumption was wrong. Compare against your current
Fig S2G before relying on this.

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


# %% Load data and select recordings

set_up_figure_format()

DATA_PATH = 'Data/av_and_probcond_raster.npy'
data = np.load(DATA_PATH, allow_pickle=True).item()
trmtx = data['trmtx']
raster_all_data = data['raster']
t_raster = data['t_raster']
ls_sess = trmtx['sessid'].unique()

ls_loc = ['DS', 'NAc_lat', 'BLA', 'mPFC', 'NAc_c']

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

# Crop raster and adjust timing
t_raster += 0.125
crop_win = [-0.5, 3.5]
raster_sub = raster_sub[:, (t_raster > crop_win[0]) & (t_raster < crop_win[1])]
t_raster = t_raster[(t_raster > crop_win[0]) & (t_raster < crop_win[1])]

# Some fixing for DA4fib01
trmtx_sub['loc'] = trmtx_sub['loc'].replace({'NAc_c': 'NAc_lat'})
ls_loc = ['DS', 'NAc_lat', 'BLA', 'mPFC']
ls_sess = trmtx_sub['sessid'].unique()

trmtx_sub['grab_reinf'] = np.mean(raster_sub[:, (t_raster > 1.5) & (t_raster < 2.5)], axis=1)


# %% Correlation between this trial's dopamine and the next trial's
# anticipatory licking, per session x region (cued trials only, so that
# "anticipatory licking" is a meaningful measure on both sides of the pair)

rng = np.random.default_rng(25)
results_corr = []

for sess in ls_sess:
    for loc in ls_loc:
        sub = trmtx_sub[(trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == loc)].sort_values('Trial#')
        if len(sub) < 2:
            continue

        tone_id = sub['ToneID'].to_numpy()
        grab_reinf = sub['grab_reinf'].to_numpy()
        anticip = sub['Anticipatory(tUS-CS) l/s'].to_numpy()

        cued_pre = tone_id[:-1] == 2
        cued_post = tone_id[1:] == 2
        grab_pre = grab_reinf[:-1]
        anticip_post = anticip[1:]

        idx_pair = cued_pre & cued_post
        if idx_pair.sum() < 3:
            continue

        grab_pre_c = grab_pre[idx_pair]
        anticip_post_c = anticip_post[idx_pair]

        r, p = stats.pearsonr(grab_pre_c, anticip_post_c)

        # Shuffled control
        anticip_sh = anticip_post_c.copy()
        rng.shuffle(anticip_sh)
        r_sh, _ = stats.pearsonr(grab_pre_c, anticip_sh)

        results_corr.append({'sess': sess, 'loc': loc, 'pearsonr': r, 'pval': p, 'pearsonr_sh': r_sh})

results_corr = pd.DataFrame(results_corr)
results_corr['loc'] = pd.Categorical(results_corr['loc'], categories=ls_loc, ordered=True)


# %% Fig S2G

fig, ax = plt.subplots(figsize=(2.2, 3))
a = ax
sns.lineplot(results_corr, x='loc', y='pearsonr_sh', hue='loc',
             errorbar='se', err_style='bars', marker='o', markeredgecolor='k',
             palette=['w'], err_kws={'ecolor': 'w', 'linewidth': 0.5}, ax=a, legend=False)
sns.lineplot(results_corr, x='loc', y='pearsonr', hue='loc',
             errorbar='se', err_style='bars', marker='o', ax=a, legend=False)
a.axhline(0, lw=0.5, ls=':', alpha=0.5, color='k')
a.set_xlabel('')
a.set_ylabel('Correlation\nGRAB pre vs anticip post')
fig.tight_layout()
