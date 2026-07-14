#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:31:20 2026

@author: vbp

Figure S3: larger time-window version of the region-specific dopamine
analysis (panels A-E), including baseline drift across trial types and
across session time.

Data can be downloaded here: https://doi.org/10.17605/OSF.IO/DV724
"""

from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score


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


# %% Plotting helpers

def plot_raster(data, x=None, ax=None, vmin=None, vmax=None, cmap=None):
    """Plot a trial x time heatmap (raster) on the given axis."""
    if x is None:
        x = np.arange(data.shape[1])
    if ax is None:
        ax = plt.gca()
    if cmap is None:
        cmap = sns.color_palette('mako', as_cmap=True)

    yaxis = np.arange(len(data))
    ax.pcolormesh(x, yaxis, data, cmap=cmap, vmin=vmin, vmax=vmax)
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


def auroc_two_samples(A, B):
    """
    Returns AUROC treating B as the 'positive' group and larger values as
    evidence for B > A.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    y_true = np.r_[np.zeros(len(A)), np.ones(len(B))]   # 0 = A, 1 = B
    y_score = np.r_[A, B]                               # scores are the raw values
    return roc_auc_score(y_true, y_score)


# %% Stats: repeated-measures ANOVA with paired Holm post-hocs

def test_rmanova_by_loc(df, id_col="sess", loc_col="loc", dv_col="delta_punish", mc_method="holm"):
    """
    Repeated-measures ANOVA of `dv_col` across `loc_col`, with subject
    (`id_col`) as a fixed effect, plus paired post-hoc comparisons between
    every pair of locations (Holm-corrected) and model-adjusted means (EMMs).
    """
    d = df[[id_col, loc_col, dv_col]].dropna().copy()
    d[id_col] = d[id_col].astype('category')
    d[loc_col] = d[loc_col].astype('category')

    # OLS with subject (session) fixed effects
    ols = smf.ols(f"{dv_col} ~ C({loc_col}) + C({id_col})", data=d).fit()
    a2 = sm.stats.anova_lm(ols, typ=2)

    # Partial eta^2
    ss_eff = float(a2.loc[f"C({loc_col})", "sum_sq"])
    ss_err = float(a2.loc["Residual", "sum_sq"])
    eta_p2 = ss_eff / (ss_eff + ss_err)

    # Paired post-hocs: only sessions that contribute both levels for each pair
    wide = d.pivot_table(index=id_col, columns=loc_col, values=dv_col, aggfunc='mean')
    levels = list(wide.columns)
    rows = []
    for a, b in combinations(levels, 2):
        sub = wide[[a, b]].dropna()
        if len(sub) < 2:
            continue
        x, y = sub[a].to_numpy(), sub[b].to_numpy()
        diff = y - x
        n = len(diff)
        dfree = n - 1
        m = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1))
        se = sd / np.sqrt(n)
        tval = m / se if se > 0 else np.inf * np.sign(m)
        p = 2 * stats.t.sf(np.abs(tval), dfree)
        tcrit = stats.t.ppf(0.975, dfree)
        ci_low, ci_high = m - tcrit * se, m + tcrit * se
        dz = tval / np.sqrt(n)
        sd_a, sd_b = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
        s_av = 0.5 * (sd_a + sd_b) if (sd_a + sd_b) > 0 else np.nan
        J = 1 - (3 / (4 * dfree - 1)) if dfree > 1 else 1.0
        g_av = J * (m / s_av) if s_av > 0 else np.nan
        rows.append(dict(level_a=a, level_b=b, n=n, mean_diff=m, ci_low=ci_low, ci_high=ci_high,
                          t=tval, df=dfree, p_unc=p, effsize_dz=dz, hedges_g_av=g_av))
    post = pd.DataFrame(rows)
    if not post.empty:
        post['p_corr'] = multipletests(post['p_unc'], method=mc_method)[1]
        post['reject'] = post['p_corr'] < 0.05
        post = post.sort_values('p_corr').reset_index(drop=True)

    # EMMs (model-adjusted means) per loc: average predictions across sessions that have that loc
    emms = []
    ref_loc = d[loc_col].cat.categories[0]
    for loc in levels:
        sess_with = d.loc[d[loc_col] == loc, id_col].unique()
        preds = []
        for s in sess_with:
            # Build a row for prediction: Intercept + loc + session dummies
            X = []
            for pname in ols.params.index:
                if pname == 'Intercept':
                    X.append(1.0)
                elif pname.startswith(f"C({loc_col})[T."):
                    lvl = pname.split('[T.', 1)[1][:-1]
                    X.append(1.0 if (loc != ref_loc and lvl == loc) else 0.0)
                elif pname.startswith(f"C({id_col})[T."):
                    sid = pname.split('[T.', 1)[1][:-1]
                    X.append(1.0 if sid == str(s) else 0.0)
                else:
                    X.append(0.0)
            preds.append(float(np.dot(X, ols.params.values)))
        if preds:
            m = float(np.mean(preds))
            se = float(np.std(preds, ddof=1) / np.sqrt(len(preds))) if len(preds) > 1 else np.nan
            tcrit = stats.t.ppf(0.975, len(preds) - 1) if len(preds) > 1 else np.nan
            ci_l = m - tcrit * se if len(preds) > 1 else np.nan
            ci_h = m + tcrit * se if len(preds) > 1 else np.nan
            emms.append(dict(loc=loc, mean=m, n_subjects=len(preds), ci_low=ci_l, ci_high=ci_h))
    emms = pd.DataFrame(emms)

    return dict(anova_table=a2, eta_p2=eta_p2, posthocs=post, emms=emms, levels=levels)


# %% Load data and select multi-site recordings (PFC, NAc_lat, DS, BLA)

set_up_figure_format()

DATA_PATH = 'Data/av_and_probcond_raster_woBLsub.npy'
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

# Crop raster and adjust timing (larger window than the main-figure version)
t_raster += 0.125
win = [-5, 15]
raster_sub = raster_sub[:, (t_raster > win[0]) & (t_raster < win[1])]
t_raster = t_raster[(t_raster > win[0]) & (t_raster < win[1])]

# Some fixing for DA4fib01
trmtx_sub['loc'] = trmtx_sub['loc'].replace({'NAc_c': 'NAc_lat'})
ls_loc = ['DS', 'NAc_lat', 'BLA', 'mPFC']


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
trmtx_sub['loc'] = pd.Categorical(trmtx_sub['loc'], categories=ls_loc, ordered=True)


# %% Panel A: example session raster, larger window

sess = 'DA4fib02_2024-02-26'
plt.close(sess)
fig, ax = plt.subplots(6, 4, figsize=(8, 6), sharex=True, num=sess)
for j, trial_selector in enumerate(ls_trial):
    # Sort trials smallest-to-largest by peak response in DS, for display order
    idx_trial_select = (trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == 'DS')
    idx_trial_select = idx_trial_select & (trmtx_sub['ToneID'] == trial_selector[0]) & (trmtx_sub['Reward?'] == trial_selector[1])
    idx_trial_select = idx_trial_select.to_numpy()
    ras = raster_sub[idx_trial_select]
    m = np.mean(ras[:, (t_raster > 1.5) & (t_raster < 2)], axis=1)
    idx_sort = np.argsort(m)

    for i, loc in enumerate(ls_loc):
        idx_trial_select = (trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == loc)
        if idx_trial_select.sum() > 0:
            vmin = np.percentile(raster_sub[idx_trial_select.to_numpy()], 1)
            vmax = np.percentile(raster_sub[idx_trial_select.to_numpy()], 99.9)

            idx_trial_select = idx_trial_select & (trmtx_sub['ToneID'] == trial_selector[0]) & (trmtx_sub['Reward?'] == trial_selector[1])
            idx_trial_select = idx_trial_select.to_numpy()

            ras = raster_sub[idx_trial_select]
            # TODO: this sort is currently never applied — len(idx_sort) can
            # never be -1, so trial rows are shown in their original order.
            # Fix the condition here if sorting by DS peak response was intended.
            if len(idx_sort) == -1:
                ras = ras[idx_sort]

            if len(ras) > 0:
                a = ax[j, i]
                plot_raster(ras, x=t_raster, ax=a, vmin=vmin, vmax=vmax)
                a.axvline(0, ls='--', color='w', alpha=0.5)
                a.axvline(1.5, ls='--', color='w', alpha=0.5)
                if i > 0:
                    a.set_yticklabels([])
                else:
                    a.set_ylabel(ls_ylabel[j], color=ls_color_trial[j])
                if j == 0:
                    a.set_title(loc, fontsize=9)

                a = ax[5, i]
                bounded_plot(ras, t_raster, ax=a, color=ls_color_trial[j])
                if j == 0:
                    a.axvline(0, ls='--', color='k', alpha=0.5)
                    a.axvline(1.5, ls='--', color='k', alpha=0.5)
                    if i == 0:
                        a.set_ylabel('GRAB-DA (z-score)')
                        a.set_xlabel('Time from cue (s)')
a.set_xlim(-5, 15)  # applies to all panels since sharex=True
fig.tight_layout()


# %% Panel B: session-averaged rasters per location x trial type

gidx = trmtx_sub.groupby(['sessid', 'loc', 'trial_type'], observed=True).indices

ls_ylabel = ['Uncued reward', 'Cued reward', 'Omission', 'Air puff', 'Omission & air puff']
ls_col_loc = ["#ff595e", "#ff924c", "#52a675", "#4267ac"]

labels = []
means = []
for key, idx in gidx.items():
    labels.append(key)
    means.append(np.nanmean(raster_sub[np.fromiter(idx, dtype=int)], axis=0))

ras_sess_averages = np.stack(means)  # shape: (n_groups, n_timepoints)
idx = pd.DataFrame(labels, columns=['an', 'loc', 'trial_type'])
# labels is a list of (sessid, loc, trial_type)

ls_loc = ['DS', 'NAc_lat', 'BLA', 'mPFC']

# NOTE: grid is 5 rows for 4 locations, leaving one blank row — intentional
# spacing, or change to plt.subplots(len(ls_loc), len(ls_trial_type)) if not.
fig, ax = plt.subplots(5, 5, figsize=(8, 6), sharex=True)
for i, loc in enumerate(ls_loc):
    for j, trial in enumerate(ls_trial_type):
        idx_select = (idx['loc'] == loc) & (idx['trial_type'] == trial)

        # Sort order computed once from 'uncued rew', then reused for all
        # trial types below.
        # NOTE: this assumes every session that contributes an 'uncued rew'
        # row for this location also contributes a row for every other trial
        # type (same count, same session order). If any session is missing a
        # trial type at this location, idx_sort could mismatch or index out
        # of bounds for that trial type's subset.
        if j == 0:
            X = ras_sess_averages[idx_select]
            m = np.mean(X[:, (t_raster > 1.5) & (t_raster < 2.5)], axis=1)
            idx_sort = np.argsort(m)

        a = ax[i, j]
        plot_raster(ras_sess_averages[idx_select][idx_sort], x=t_raster, ax=a, vmin=-1.5, vmax=5, cmap='gray_r')

        if j == 0:
            a.set_ylabel(loc, color=ls_col_loc[i])
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
        bounded_plot(ras_sess_averages[idx_select], x=t_raster, ax=a, color=ls_col_loc[i])
        if i == 0 and j == 0:
            a.set_ylabel('GRAB-DA\n(z-score)')
            a.set_xlabel('Time from cue (s)')
        a.set_ylim(-2.5, 7)
fig.tight_layout()


# %% Trial-duration (ITI) histogram

iti_chunks = []
for sess in ls_sess:
    loc = trmtx_sub[trmtx_sub['sessid'] == sess]['loc'].unique()
    iti_real = np.diff(trmtx_sub[(trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == loc[0])]['timeTrialStart'])
    iti_chunks.append(iti_real)
all_iti = np.concatenate(iti_chunks)

fig, ax = plt.subplots(figsize=(3, 3))
ax.hist(all_iti, np.linspace(0, 50, 100), color='gray', lw=0.25)
ax.axvline(np.mean(all_iti), color='k', lw='0.25', alpha=0.5)
ax.set_ylabel('Count')
ax.set_xlabel('Trial duration (s)')
ax.set_title(f'Avg = {np.mean(all_iti):0.2f} s')
fig.tight_layout()


# %% Baseline and post-stim (14-15s) mean per trial, per session/location

trmtx_sub['grab_bl'] = np.mean(raster_sub[:, (t_raster > -1) & (t_raster < 0)], axis=1)
trmtx_sub['grab_reinf'] = np.mean(raster_sub[:, (t_raster > 14) & (t_raster < 15)], axis=1)

ls_comp = ['uncued rew', 'cued rew', 'uncued pun', 'omission pun']

results = []
for comp in ls_comp:
    for sess in ls_sess:
        for loc in ls_loc:
            idx_select = (trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == loc)
            if idx_select.sum() == 0:
                continue

            A = trmtx_sub['grab_reinf'][idx_select & (trmtx_sub['trial_type'] == comp)].to_numpy()
            B = trmtx_sub['grab_bl'][idx_select & (trmtx_sub['trial_type'] == comp)].to_numpy()

            results.append({
                'sess': sess,
                'loc': loc,
                'comp': f'{comp} vs baseline',
                'auroc': auroc_two_samples(B, A),
                'mean_val': np.mean(A) - np.mean(B),
                'mean_bl': np.mean(B),
            })

results = pd.DataFrame(results)

all_comp_labels = results['comp'].unique()
ls_cond_vsbl = [x for x in all_comp_labels if 'vs baseline' in x]


# %% Panel C: baseline compared across trial type, per location

fig, ax = plt.subplots(1, len(ls_loc), figsize=(4, 4), sharey=True)
for i, loc in enumerate(ls_loc):
    a = ax[i]
    sub = results.loc[results['loc'] == loc]

    sns.lineplot(sub, x='comp', y='mean_bl', estimator=None, units='sess',
                 ax=a, color=ls_col_loc[i], alpha=0.3, linewidth=0.25, marker='o', markersize=2)
    sns.lineplot(sub, x='comp', y='mean_bl', errorbar='se',
                 ax=a, color=ls_col_loc[i], err_style='bars', marker='o', markersize=5)
    a.set_ylim(-1.5, 1.5)
    a.set_ylabel('Baseline GRAB-DA (z-score)')
    a.set_xlabel('')
    a.set_xlim(-0.5, 3.5)
    plt.setp(a.get_xticklabels(), rotation=90, ha='right')

    anova_res = test_rmanova_by_loc(sub, id_col='sess', loc_col='comp', dv_col='mean_bl')
    pval = anova_res['anova_table']['PR(>F)']['C(comp)']
    a.set_title(f'{loc}\n{pval:0.3e}')

    print()
    print(anova_res['posthocs'])

fig.tight_layout()


# %% Panel D, E: baseline progression across session time

all_pears = []
for sess in ls_sess:
    if sess == 'DA4fib06_2024-05-11':
        fig, ax = plt.subplots(figsize=(3, 3))
    for j, loc in enumerate(ls_loc):
        idx = (trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == loc)
        if np.sum(idx) == 0:
            continue

        bl = trmtx_sub[idx]['grab_bl']
        t = trmtx_sub[idx]['timeTone']
        bl_smooth = np.convolve(bl, np.ones(20) / 20, mode='same')

        r, p = stats.pearsonr(t, bl)
        all_pears.append({'sess': sess, 'loc': loc, 'pearsonr': r, 'p-val': p})

        if sess == 'DA4fib06_2024-05-11':
            ax.plot(t, bl, 'o', markersize=1, color=ls_col_loc[j], label=loc)
            ax.plot(t, bl_smooth, color=ls_col_loc[j], lw=1)
    if sess == 'DA4fib06_2024-05-11':
        ax.set_title(sess)
        ax.set_xlabel('Time from session start (s)')
        ax.set_ylabel('Baseline GRAB-DA (z-score)')
        ax.legend()
        fig.tight_layout()

all_pears = pd.DataFrame(all_pears)

fig, ax = plt.subplots(figsize=(2, 3))
a = ax
sns.swarmplot(all_pears, x='loc', y='pearsonr', palette=ls_col_loc)
a.axhline(0, ls='--', color='k', alpha=0.5)
a.set_ylim(-.5, .5)
a.set_ylabel('Baseline GRAB-DA vs time (pearson-r)')
a.set_xlabel('')
a.set_xlim(-0.5, 3.5)
plt.setp(a.get_xticklabels(), rotation=90, ha='right')
fig.tight_layout()
