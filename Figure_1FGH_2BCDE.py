
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 08:11:52 2025

@author: vbp

Batch analysis for Figures 1F, 1G, 1H and 2B, 2C, 2D, 2E.
Sorts files from each target into subfolders and plots per-subfolder data.

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

def plot_raster(data, x=None, ax=None, vmin=None, vmax=None):
    """Plot a trial x time heatmap (raster) on the given axis."""
    if x is None:
        x = np.arange(data.shape[1])
    if ax is None:
        ax = plt.gca()

    yaxis = np.arange(len(data))
    ax.pcolormesh(x, yaxis, data,
                  cmap=sns.color_palette('mako', as_cmap=True),
                  vmin=vmin, vmax=vmax)
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


# %% auROC computation helper

def compute_auroc_results(df, ls_loc, comps, vs_baseline=False,
                           value_col='grab_reinf', cue_col='grab_cue'):
    """
    Compute per-session, per-location auROC values.

    If vs_baseline=False, `comps` is a list of [condition_A, condition_B]
    pairs; auROC treats condition_B as evidence for the 'positive' class.
    If vs_baseline=True, `comps` is a list of single condition names, each
    compared against a zero baseline. The special value 'cs+' compares
    `cue_col` (not `value_col`) for 'cued rew' trials against baseline.
    """
    rows = []
    for comp in comps:
        for sess in df['sessid'].unique():
            for loc in ls_loc:
                idx = (df['sessid'] == sess) & (df['loc'] == loc)
                if idx.sum() == 0:
                    continue

                if vs_baseline:
                    if comp == 'cs+':
                        A = df[cue_col][idx & (df['trial_type'] == 'cued rew')].to_numpy()
                    else:
                        A = df[value_col][idx & (df['trial_type'] == comp)].to_numpy()
                    B = np.zeros(A.shape)
                    label = f'{comp} vs baseline'
                else:
                    A = df[value_col][idx & (df['trial_type'] == comp[0])].to_numpy()
                    B = df[value_col][idx & (df['trial_type'] == comp[1])].to_numpy()
                    label = f'{comp[0]} vs {comp[1]}'

                rows.append({
                    'sess': sess,
                    'loc': loc,
                    'comp': label,
                    'auroc': auroc_two_samples(B, A),
                    'mean_val': np.mean(A) - np.mean(B),
                })
    return rows


def plot_auroc_comparisons(conditions, titles, results, ls_loc, ylabel):
    """
    Plot one auROC panel per condition (used for both Fig 1H and Fig 2C):
    per-session lines + mean +/- SEM, a repeated-measures ANOVA across
    locations (title shows the p-value), and a paired t-test of each
    location's value against baseline (printed, not plotted).
    """
    fig, ax = plt.subplots(1, len(conditions), figsize=(6, 3), sharey=True)
    for i, cond in enumerate(conditions):
        a = ax[i]
        sub = results.loc[results['comp'] == cond]

        sns.lineplot(sub, x='loc', y='auroc', estimator=None, units='sess',
                     ax=a, color='k', alpha=0.3, linewidth=0.25, marker='o', markersize=2)
        sns.lineplot(sub, x='loc', y='auroc', errorbar='se',
                     ax=a, color='k', err_style='bars', marker='o', markersize=5)
        a.axhline(0.5, ls='--', color='k', alpha=0.5)
        a.set_ylim(-0.05, 1.05)
        a.set_ylabel(ylabel)
        a.set_xlabel('')
        a.set_xlim(-0.5, 3.5)
        plt.setp(a.get_xticklabels(), rotation=90, ha='right')

        # Repeated-measures ANOVA across locations (session as fixed effect)
        anova_res = test_rmanova_by_loc(sub, id_col='sess', loc_col='loc', dv_col='auroc')
        pval = anova_res['anova_table']['PR(>F)']['C(loc)']
        a.set_title(f'{titles[i]}\n{pval:0.3e}')

        print(cond)
        print()
        print(anova_res['posthocs'])

        # Paired t-test of each location's value against baseline
        ttest_res = []
        for loc in ls_loc:
            vals = results[(results['comp'] == cond) & (results['loc'] == loc)]['mean_val']
            s, p = stats.ttest_rel(vals, np.zeros(vals.shape))
            ttest_res.append({'loc': loc, 't-stat': s, 'p': p, 'p_corr': p * 4})
        print(pd.DataFrame(ttest_res))

    fig.tight_layout()
    return fig, ax


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

# Crop raster and adjust timing
t_raster += 0.125
win = [-0.5, 10.5]
raster_sub = raster_sub[:, (t_raster > win[0]) & (t_raster < win[1])]
t_raster = t_raster[(t_raster > win[0]) & (t_raster < win[1])]

# Some fixing for DA4fib01
trmtx_sub['loc'] = trmtx_sub['loc'].replace({'NAc_c': 'NAc_lat'})
ls_loc = ['DS', 'NAc_lat', 'BLA', 'mPFC']


# %% Figure 1F, 1G

ls_sess = trmtx_sub['sessid'].unique()

ls_color_trial = ['dodgerblue', 'lightseagreen', 'grey', 'crimson', 'slateblue']
ls_trial = [[3, 5], [2, 5], [2, 0], [3, -1], [2, -1]]
ls_trial_type = ['uncued rew', 'cued rew', 'omission', 'uncued pun', 'omission pun']
ls_ylabel = ['Uncued\nreward', 'Cued\nreward', 'Omission', 'Air puff', 'Omission &\n air puff']

sess = 'DA4fib02_2024-02-26'
plt.close(sess)
fig, ax = plt.subplots(6, 4, figsize=(6, 6), sharex=True, num=sess)
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
            # vmin/vmax from all trials at this location (shared scale across trial types)
            vmin = np.percentile(raster_sub[idx_trial_select.to_numpy()], 1)
            vmax = np.percentile(raster_sub[idx_trial_select.to_numpy()], 99.9)

            idx_trial_select = idx_trial_select & (trmtx_sub['ToneID'] == trial_selector[0]) & (trmtx_sub['Reward?'] == trial_selector[1])
            idx_trial_select = idx_trial_select.to_numpy()

            ras = raster_sub[idx_trial_select]

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
a.set_xlim(-0.5, 3.5)  # applies to all panels since sharex=True
fig.tight_layout()


# %% Add trial type labels and per-trial mean cue/reinforcement values

tr_type = np.zeros(len(trmtx_sub), dtype=object)
for j, trial_selector in enumerate(ls_trial):
    tr_type[(trmtx_sub['ToneID'] == trial_selector[0]) & (trmtx_sub['Reward?'] == trial_selector[1])] = ls_trial_type[j]
tr_type[tr_type == 0] = 'NA'
trmtx_sub['trial_type'] = tr_type

trmtx_sub['trial_type'] = pd.Categorical(trmtx_sub['trial_type'], categories=ls_trial_type, ordered=True)
trmtx_sub['loc'] = pd.Categorical(trmtx_sub['loc'], categories=ls_loc, ordered=True)

trmtx_sub['grab_cue'] = np.mean(raster_sub[:, (t_raster > 0) & (t_raster < 1)], axis=1)
trmtx_sub['grab_reinf'] = np.mean(raster_sub[:, (t_raster > 1.5) & (t_raster < 2.5)], axis=1)

# Mean grouped by session, location, and trial type
mean_sess = trmtx_sub.groupby(['sessid', 'loc', 'trial_type'])[['grab_cue', 'grab_reinf']].mean()


# %% Figure 2B

ls_xticks = ['Uncued rew.', 'Cued reward', 'Omission', 'Air puff', 'Omission & air']
fig, ax = plt.subplots(figsize=(3, 3))
a = ax
sns.lineplot(mean_sess, x='trial_type', y='grab_reinf', hue='loc',
             err_style='bars', marker='o', lw=1, ax=a)
a.axhline(0, ls='--', color='k', alpha=0.5)
a.set_xlabel('')
a.set_ylabel('GRAB-DA amplitude (z-score)')
a.set_xticklabels(ls_xticks, rotation=90)
a.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
fig.tight_layout()


# %% Compute auROC tables (pairwise comparisons + vs. baseline)

ls_comp_pairs = [
    ['uncued rew', 'cued rew'],
    ['uncued pun', 'omission'],
    ['uncued pun', 'omission pun'],
    ['uncued rew', 'uncued pun'],
]
ls_cond_baseline = ['cs+', 'uncued rew', 'cued rew', 'omission', 'uncued pun', 'omission pun']

results = []
results += compute_auroc_results(trmtx_sub, ls_loc, ls_comp_pairs, vs_baseline=False)
results += compute_auroc_results(trmtx_sub, ls_loc, ls_cond_baseline, vs_baseline=True)
results = pd.DataFrame(results)

all_comp_labels = results['comp'].unique()
ls_cond_vsbl = [x for x in all_comp_labels if 'vs baseline' in x]
ls_cond_dual = [x for x in all_comp_labels if 'vs baseline' not in x]


# %% Figure 1H

titles_1h = ['Cue', 'Uncued\nreward', 'Cued\nreward', 'Omission', 'Air puff', 'Omission &\nAir puff']
fig, ax = plot_auroc_comparisons(ls_cond_vsbl, titles_1h, results, ls_loc, ylabel='auROC (vs baseline)')


# %% Figure 2C

titles_2c = ['Uncued vs.\ncued reward', 'Air puff vs\nomission', 'Air puff vs\nair & omission', 'Reward vs\nair puff']
fig, ax = plot_auroc_comparisons(ls_cond_dual, titles_2c, results, ls_loc, ylabel='auROC')


# %% Figure 2D, 2E — scatter plots of auROC comparisons against each other

# Pivot to one row per (session, location) with one column per comparison.
# Equivalent to the original nested-loop lookup, but avoids rebuilding the
# lookup by hand for every (sess, loc, comp) combination.
results_scatterable = (
    results.pivot_table(index=['sess', 'loc'], columns='comp', values='auroc')
    .reset_index()
)

ls_scatter_pairs = [
    ['omission vs baseline', 'uncued pun vs baseline'],
    ['uncued rew vs cued rew', 'uncued pun vs baseline'],
]
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
for i, cond in enumerate(ls_scatter_pairs):
    a = ax[i]
    sns.scatterplot(results_scatterable, x=cond[0], y=cond[1], hue='loc', ax=a)
    a.set_xlim(-0.05, 1.05)
    a.set_ylim(-0.05, 1.05)
    a.axhline(0.5, ls='--', color='k', alpha=0.5)
    a.axvline(0.5, ls='--', color='k', alpha=0.5)

    r, p = stats.pearsonr(results_scatterable[cond[0]], results_scatterable[cond[1]])
    a.set_title(f'r={r:0.2f} p={p:0.3f}')
fig.tight_layout()