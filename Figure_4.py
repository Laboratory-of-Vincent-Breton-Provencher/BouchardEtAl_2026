#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 09:54:13 2025

@author: vbp

Reward-magnitude analysis for Figure 4.

    Fig 4A, 4B - example session raster + average traces, per region x
                 reward size (0.3, 1, 2.5, 5, 10 uL)
    Fig 4C     - GRAB-DA amplitude vs. reward size, per region
    Fig 4D     - auROC (0.3 uL vs 10 uL), per region
    Fig 4E     - correlation (Pearson r) between reward size and GRAB-DA
                 amplitude, per session/region
    Fig 4F     - that correlation vs. beta_R (from the regression model)

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
    ax.pcolormesh(x, yaxis, data, cmap=sns.color_palette('mako', as_cmap=True), vmin=vmin, vmax=vmax)
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
    y_score = np.r_[A, B]                                # scores are the raw values
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

    ols = smf.ols(f"{dv_col} ~ C({loc_col}) + C({id_col})", data=d).fit()
    a2 = sm.stats.anova_lm(ols, typ=2)

    ss_eff = float(a2.loc[f"C({loc_col})", "sum_sq"])
    ss_err = float(a2.loc["Residual", "sum_sq"])
    eta_p2 = ss_eff / (ss_eff + ss_err)

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

    emms = []
    ref_loc = d[loc_col].cat.categories[0]
    for loc in levels:
        sess_with = d.loc[d[loc_col] == loc, id_col].unique()
        preds = []
        for s in sess_with:
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

data = np.load('Data/rewmag_raster.npy', allow_pickle=True).item()
trmtx = data['trmtx']
raster_all_data = data['raster']
t_raster = data['t_raster']
ls_sess = trmtx['sessid'].unique()

ls_loc = ['DS', 'NAc_lat', 'BLA', 'mPFC', 'NAc_c']
# ls_loc = ['DS','NAc_lat','BLA','mPFC','OT','NAc_c','NAc_m','TS']

idx_select = np.zeros((len(trmtx)), dtype=bool)
for sess in ls_sess:
    sub_loc = trmtx['loc'][trmtx['sessid'] == sess].unique()
    if len(sub_loc) > 2:
        if set(sub_loc).issubset(ls_loc):
            idx_select = idx_select | ((trmtx['sessid'] == sess).to_numpy())
trmtx_sub = trmtx.iloc[idx_select, :].copy()
raster_sub = raster_all_data[idx_select, :]

# Crop raster and adjust timing (+1.5 to center on reinforcement)
t_raster += 0.125 - 1.5
crop_win = [-0.5, 3.5]
raster_sub = raster_sub[:, (t_raster > crop_win[0]) & (t_raster < crop_win[1])]
t_raster = t_raster[(t_raster > crop_win[0]) & (t_raster < crop_win[1])]

# Some fixing for DA4fib01
trmtx_sub['loc'] = trmtx_sub['loc'].replace({'NAc_c': 'NAc_lat'})
ls_loc = ['DS', 'NAc_lat', 'BLA', 'mPFC']
ls_sess = trmtx_sub['sessid'].unique()


# %% Trial (reward size) definitions

ls_color_trial = sns.color_palette('mako_r', 5)
ls_trial = [[3, 0.3], [3, 1], [3, 2.5], [3, 5], [3, 10]]
ls_trial_type = [0.3, 1, 2.5, 5, 10]
ls_ylabel = ['0.3 uL', '1 uL', '2.5 uL', '5 uL', '10 uL']


# %% Fig 4A, 4B: example session raster + average traces

example_sess = 'DA4fib03_2024-02-20'

plt.close(example_sess)
fig, ax = plt.subplots(6, 4, figsize=(9, 6), num=example_sess)
for j, trial_selector in enumerate(ls_trial):
    for i, loc in enumerate(ls_loc):
        idx_trial_select = (trmtx_sub['sessid'] == example_sess) & (trmtx_sub['loc'] == loc)
        if idx_trial_select.sum() == 0:
            continue

        vmin = np.percentile(raster_sub[idx_trial_select.to_numpy()], 1)
        vmax = np.percentile(raster_sub[idx_trial_select.to_numpy()], 99.9)

        idx_trial_select = idx_trial_select & (trmtx_sub['ToneID'] == trial_selector[0]) & (trmtx_sub['Reward?'] == trial_selector[1])
        idx_trial_select = idx_trial_select.to_numpy()
        ras = raster_sub[idx_trial_select]
        if len(ras) == 0:
            continue

        a = ax[j, i]
        plot_raster(ras, x=t_raster, ax=a, vmin=vmin, vmax=vmax)
        a.axvline(0, ls='--', color='w', alpha=0.5)
        if i > 0:
            a.set_yticklabels([])
        else:
            a.set_ylabel(ls_ylabel[j], color=ls_color_trial[j])
        if j == 0:
            a.set_title(f"{loc} ({vmin:1.1f} {vmax:1.1f})", fontsize=9)
        a.set_xlim(crop_win[0], 2)
        a.set_xticklabels('')

        a = ax[5, i]
        bounded_plot(ras, t_raster, ax=a, color=ls_color_trial[j])
        if j == 0:
            a.axvline(0, ls='--', color='k', alpha=0.5)
            if i == 0:
                a.set_ylabel('GRAB-DA (z-score)')
                a.set_xlabel('Time from cue (s)')
            a.set_xlim(crop_win[0], 2)
fig.tight_layout()


# %% Label trial types and compute post-reward mean response

tr_type = np.zeros(len(trmtx_sub), dtype=object)
for j, trial_selector in enumerate(ls_trial):
    tr_type[(trmtx_sub['ToneID'] == trial_selector[0]) & (trmtx_sub['Reward?'] == trial_selector[1])] = ls_trial_type[j]
tr_type[tr_type == 0] = np.nan
trmtx_sub['trial_type'] = tr_type
trmtx_sub['trial_type'] = pd.Categorical(trmtx_sub['trial_type'], categories=ls_trial_type, ordered=True)
trmtx_sub['loc'] = pd.Categorical(trmtx_sub['loc'], categories=ls_loc, ordered=True)

trmtx_sub['grab_reinf'] = np.mean(raster_sub[:, (t_raster > 0) & (t_raster < 1)], axis=1)

# Mean amplitude per session x region x reward size (Fig 4C)
mean_sess = trmtx_sub.groupby(['sessid', 'loc', 'trial_type'])[['grab_reinf']].mean()


# %% Fig 4D: auROC (0.3 uL vs 10 uL), per region

comp = (10, 0.3)  # (high, low) reward sizes compared

results_auroc = []
for sess in ls_sess:
    for loc in ls_loc:
        idx_select = (trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == loc)
        if idx_select.sum() == 0:
            continue

        A = trmtx_sub['grab_reinf'][idx_select & (trmtx_sub['trial_type'] == comp[0])].to_numpy()
        B = trmtx_sub['grab_reinf'][idx_select & (trmtx_sub['trial_type'] == comp[1])].to_numpy()

        results_auroc.append({
            'an': trmtx_sub.loc[idx_select]['anid'].unique()[0],
            'sess': sess,
            'loc': loc,
            'comp': f'{comp[0]} vs {comp[1]}',
            'auroc': auroc_two_samples(B, A),
            'mean_val': np.mean(A) - np.mean(B),
        })

results_auroc = pd.DataFrame(results_auroc)


# %% Fig 4E: correlation (Pearson r) between reward size and GRAB-DA
# amplitude, per session x region

results_all = []
for sess in ls_sess:
    for loc in ls_loc:
        idx_trial_select = (trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == loc)
        if idx_trial_select.sum() == 0:
            continue

        x = trmtx_sub.loc[idx_trial_select]['trial_type'].to_numpy()
        y = trmtx_sub.loc[idx_trial_select]['grab_reinf'].to_numpy()
        y_real = y[~np.isnan(x)]
        x_real = x[~np.isnan(x)]

        results_all.append({
            'an': trmtx_sub.loc[idx_trial_select]['anid'].unique()[0],
            'sess': sess,
            'loc': loc,
            'pearsonr': stats.pearsonr(x_real, y_real)[0],
        })

results_all = pd.DataFrame(results_all)


# %% Paired tests: 0.3 uL vs 10 uL, per region

for loc in ls_loc:
    val = results_auroc[results_auroc['loc'] == loc]['mean_val'].to_numpy()
    print()
    print(f'{loc} {comp[0]} vs {comp[1]}')
    print('pval = {}'.format(stats.ttest_rel(val, np.zeros_like(val))[1] * 4))


# %% Fig 4F prep: correlation (Pearson r, reward size vs. response) merged
# with beta_R (from the regression model), per animal x region

# NOTE: as with the other regression scripts, double-check this path
# points at the right copy of results_multisite_regression2.csv.
best_value_fixed = pd.read_csv('Data/results_multisite_regression2.csv')

pearson_an = results_all.groupby(['an', 'loc'])['pearsonr'].mean()
bmotiv_an = best_value_fixed.groupby(['an', 'loc'])[['b_motiv', 'b_senso']].mean()

results_combo = pd.concat([pd.DataFrame(pearson_an), pd.DataFrame(bmotiv_an)], axis=1)
results_combo = results_combo.reset_index()
results_combo['loc'] = pd.Categorical(results_combo['loc'], categories=ls_loc, ordered=True)
results_combo['an_simple'] = [x[:8] for x in results_combo['an']]

# Drop animals missing either the reward-magnitude or model-regression results
results_combo = results_combo.dropna(subset=['pearsonr', 'b_motiv', 'b_senso'])


# %% Fig 4C, 4D, 4E, 4F combined

fig, ax = plt.subplots(1, 4, figsize=(9, 3))

# Fig 4C: amplitude vs. reward size, per region
a = ax[0]
sns.lineplot(mean_sess, x='trial_type', y='grab_reinf', hue='loc',
             err_style='bars', marker='o', lw=1, errorbar='se', ax=a)
a.axhline(0, ls='--', color='k', alpha=0.5)
a.set_xlabel('')
a.set_ylabel('GRAB-DA amplitude (z-score)')

# Fig 4D: auROC (0.3 uL vs 10 uL)
a = ax[1]
sns.lineplot(results_auroc, x='loc', y='auroc', estimator=None, units='sess',
             ax=a, color='k', err_style='bars', alpha=0.3, marker='o', markersize=4, legend=False)
sns.lineplot(results_auroc, x='loc', y='auroc',
             errorbar='se', err_style='bars', color='k', marker='o', ax=a)
a.set_xlabel('')
a.axhline(0.5, ls=':', color='k', alpha=0.5)
a.set_ylim(0, 1)
a.set_ylabel('auROC (0.3 uL vs 10 uL)')

statsanova = test_rmanova_by_loc(results_auroc, id_col="sess", loc_col="loc", dv_col='auroc', mc_method="holm")
p = statsanova['anova_table']['PR(>F)']['C(loc)']
a.set_title(f'p={p:0.2e}')
print('auROC:')
print(statsanova['posthocs'])

# Fig 4E: correlation (reward size vs. response), per region
# NOTE: axhline(0.5)/ylim(0,1) below were carried over from the auROC panel
# above. For auROC, 0.5 is the meaningful "chance level" reference; for a
# Pearson r, "no correlation" is 0, not 0.5. Your data happens to stay
# positive (~0.2-0.9) so ylim(0,1) isn't clipping anything, but worth
# deciding whether you want the reference line at 0 instead, or removed.
a = ax[2]
sns.lineplot(results_all, x='loc', y='pearsonr', estimator=None, units='sess',
             ax=a, color='k', err_style='bars', alpha=0.3, marker='o', markersize=4, legend=False)
sns.lineplot(results_all, x='loc', y='pearsonr',
             errorbar='se', err_style='bars', color='k', marker='o', ax=a)
a.set_xlabel('')
a.axhline(0.5, ls=':', color='k', alpha=0.5)
a.set_ylim(0, 1)
a.set_ylabel('Corr. GRAB-DA vs. reward size')

statsanova = test_rmanova_by_loc(results_all, id_col="sess", loc_col="loc", dv_col='pearsonr', mc_method="holm")
p = statsanova['anova_table']['PR(>F)']['C(loc)']
a.set_title(f'p={p:0.2e}')
print('pearsonr:')
print(statsanova['posthocs'])

# Fig 4F: that correlation vs. beta_R, colored by region
a = ax[3]
sns.lineplot(results_combo, x='pearsonr', y='b_motiv', hue='an_simple', palette=['gray'], ax=a, linewidth=0.5, legend=False)
sns.scatterplot(results_combo, x='pearsonr', y='b_motiv', hue='loc', ax=a)
r, p = stats.pearsonr(results_combo['pearsonr'], results_combo['b_motiv'])
a.set_title(f'r={r:0.2f} p={p:1.0e}')
a.set_xlabel('Corr. GRAB-DA vs. reward size')

fig.tight_layout()
