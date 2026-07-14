#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:14:05 2026

@author: vbp

Model-based analysis for Figure 3B-H and Figure S4 A, C, D, E, F, H.

Section -> panel mapping (see comments for details):
    Fig 3B            - predicted GRAB-DA, example session
    Fig 3C, 3D        - R2 score (plain, and vs. shuffled-regressor control)
    Fig 3E, 3F, 3G    - beta_S, beta_M, sensory preference index by region (+ stats)
    Fig 3H            - sensory preference index heatmap
    Fig S4A           - anticipatory lick rate vs. fitted alpha
    Fig S4C, S4D      - R2 / beta_M / beta_S heatmaps over the alpha-delta grid
    Fig S4E, S4F      - beta_S vs beta_M trajectories over the alpha-delta grid
    Fig S4H (partial) - Pearson r between R and S regressors (model side only)


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
from sklearn.linear_model import Ridge


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
    matplotlib.rcParams['image.cmap'] = 'mako'
    matplotlib.rcParams['lines.linewidth'] = 0.75

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


# %% Model parameters, trial types, grid search over alpha/delta

ls_trial = [
    [3, 5],   # Uncued reward
    [2, 5],   # Cued reward
    [2, 0],   # Reward omission
    [3, -1],  # Uncued air puff
    [2, -1],  # Reward omission & air puff
]
ls_trial_type = ['uncued rew', 'cued rew', 'omission', 'uncued pun', 'omission pun']

fit_win = [1.5, 2.5]  # window used to fit the model (relative to cue)

n_alpha = 40
n_delta = 40
ls_alpha = np.arange(n_alpha) / n_alpha
ls_delta = np.arange(n_delta + 1) / n_delta
negative_scale = 0.5

mean_post_reinf = np.mean(raster_sub[:, (t_raster > fit_win[0]) & (t_raster < fit_win[1])], axis=1)
bl = np.mean(raster_sub[:, (t_raster > fit_win[0] - 0.25) & (t_raster < fit_win[0])], axis=1)
mean_post_reinf -= bl

# NOTE: this grid search fits one Ridge model per (alpha, delta, session,
# region) combination — with the defaults above that's 40*41*n_sess*4 fits,
# and each combination prints a progress line, so this step is slow and
# verbose. Comment out the print() below to quiet it, or reduce n_alpha/
# n_delta for a faster (coarser) search.
results = []
for alpha in ls_alpha:
    motivational = encode_motivational(trmtx_sub, alpha, negative_scale=negative_scale)
    for delta in ls_delta:
        sensory = encode_sensory(trmtx_sub, delta)

        for sess in ls_sess:
            print(f'{sess}: alpha={alpha} delta={delta}')

            for loc in ls_loc:
                idx_trial_select = (trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == loc)
                if idx_trial_select.sum() == 0:
                    continue
                idx_trial_select = idx_trial_select.to_numpy()

                y = mean_post_reinf[idx_trial_select]
                X = np.array([motivational[idx_trial_select], sensory[idx_trial_select]]).T

                clf = Ridge(alpha=0)
                clf.fit(X, y)

                results.append({
                    'sess': sess,
                    'loc': loc,
                    'alpha_learning': alpha,
                    'delta_punish': delta,
                    'b0': clf.intercept_,
                    'b_motiv': clf.coef_[0],
                    'b_senso': clf.coef_[1],
                    'r2_score': clf.score(X, y),
                })

results = pd.DataFrame(results)
results['senso_bias'] = (results['b_senso'].abs() - results['b_motiv'].abs()) / (results['b_senso'].abs() + results['b_motiv'].abs())
results['loc'] = pd.Categorical(results['loc'], categories=ls_loc, ordered=True)

# Best (alpha, delta) per session x region, by R2
best_value = results.loc[results.groupby(['sess', 'loc'],observed=True)['r2_score'].idxmax()].reset_index(drop=True)
best_value['loc'] = pd.Categorical(best_value['loc'], categories=ls_loc, ordered=True)

# Add trial type, easier for plotting later
tr_type = np.zeros(len(trmtx_sub), dtype=object)
for j, trial_selector in enumerate(ls_trial):
    tr_type[(trmtx_sub['ToneID'] == trial_selector[0]) & (trmtx_sub['Reward?'] == trial_selector[1])] = ls_trial_type[j]
tr_type[tr_type == 0] = 'NA'
trmtx_sub['trial_type'] = tr_type
trmtx_sub['trial_type'] = pd.Categorical(trmtx_sub['trial_type'], categories=ls_trial_type, ordered=True)
trmtx_sub['loc'] = pd.Categorical(trmtx_sub['loc'], categories=ls_loc, ordered=True)

trmtx_sub['grab_cue'] = np.mean(raster_sub[:, (t_raster > 0) & (t_raster < 1)], axis=1)
trmtx_sub['grab_reinf'] = np.mean(raster_sub[:, (t_raster > 1.5) & (t_raster < 2.5)], axis=1)


# %% Fig S4 C, S4 D: R2 / beta_motiv / beta_senso heatmaps over the alpha-delta grid

alpha_grid = np.round(ls_alpha, 6)
delta_grid = np.round(ls_delta, 6)

fig, ax = plt.subplots(3, len(ls_loc), figsize=(10, 8), sharex=True, sharey=True)
for j, var in enumerate(['r2_score', 'b_motiv', 'b_senso']):
    pt = results.pivot_table(
        index=['loc', 'alpha_learning'], columns='delta_punish', values=var, aggfunc='mean'
    ).reindex(
        pd.MultiIndex.from_product([ls_loc, alpha_grid], names=['loc', 'alpha_learning'])
    ).reindex(columns=delta_grid)

    for i, loc in enumerate(ls_loc):
        a = ax[j, i]
        cmap = 'rocket'
        if var == 'r2_score':
            vminmax = [0, 0.8]
        else:
            vminmax = [0, 3]

        sns.heatmap(pt.loc[loc], ax=a, vmin=vminmax[0], vmax=vminmax[1], cmap=cmap, cbar=False)
        a.set_title(f'{ls_loc[i]} ({var})')
        a.set_xticks([0, len(ls_delta)])
        a.set_yticks([0, len(ls_alpha)])
        a.set_xticklabels([0, 1])
        a.set_yticklabels([0, 1])

fig.tight_layout()


# %% Figure 3H - Sensory preference index heatmap over the alpha-delta grid

fig, ax = plt.subplots(2, 2, figsize=(3, 3), sharex=True, sharey=True)
ax = ax.flatten()

pt = results.pivot_table(
    index=['loc', 'alpha_learning'], columns='delta_punish', values='senso_bias', aggfunc='mean'
).reindex(
    pd.MultiIndex.from_product([ls_loc, alpha_grid], names=['loc', 'alpha_learning'])
).reindex(columns=delta_grid)

for i, loc in enumerate(ls_loc):
    a = ax[i]
    a.pcolormesh(ls_delta, ls_alpha, pt.loc[loc].to_numpy(), vmin=-0.5, vmax=0.5, cmap='vlag')
    a.set_xticks([0, 1])
    a.set_yticks([0, 1])
    if i == 2:
        a.set_xlabel('δ')
        a.set_ylabel('ɑ')
    a.set_title(f'{ls_loc[i]}', fontsize=6)

fig.tight_layout()


# %% Fig S4 E, S4 F: beta_S vs beta_M trajectories over the alpha-delta grid

summary = results.groupby(['loc', 'alpha_learning', 'delta_punish'])[['b_senso', 'b_motiv']].mean(numeric_only=True)

# S4 E: single trajectory across all delta, colored by region
fig, ax = plt.subplots(figsize=(3, 3))
a = ax
a.axline((0.8, 0.8), (1, 1), lw=0.5, color='k', ls=':', alpha=0.5)
sns.scatterplot(summary, x='b_senso', y='b_motiv', s=4, hue='loc', lw=0.25, alpha=0.5, ax=a)
a.set_xlabel('β_S')
a.set_ylabel('β_M')
fig.tight_layout()

# S4 F: trajectories at fixed alpha, colored by delta
# NOTE: PDF shows only 2 panels ("ɑ=0.25" and "ɑ=0.75"); this loops over 3
# values (0.25, 0.5, 0.75). 
palette = sns.color_palette('gray', n_colors=6)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 3))
for i, alpha in enumerate([0.25, 0.5, 0.75]):
    a = ax[i]
    for j, delta in enumerate(np.arange(6) / 5):
        line_data = np.array([summary.loc[loc, alpha, delta].to_numpy() for loc in ls_loc]).T
        a.plot(line_data[0], line_data[1], color=palette[j], label=delta, lw=1)
        for loc in ls_loc:
            a.scatter(*summary.loc[loc, alpha, delta].to_numpy(), edgecolor=None, c='k', s=10)

    a.axline((0.8, 0.8), (1, 1), lw=0.5, color='k', ls=':', alpha=0.5)
    a.set_title(f'ɑ={alpha}')
    if i == 2:
        a.legend(title='δ')
    if i == 0:
        a.set_xlabel('β_S')
        a.set_ylabel('β_M')

fig.tight_layout()


# %% Refit with a single (alpha, delta) per session (averaged across
# regions), separately for each region, to get the final beta weights.

best_alpha_delta = best_value.groupby('sess')[['alpha_learning', 'delta_punish']].mean()
ls_sess = best_alpha_delta.index.to_numpy()

results_fixed_all = []
for sess in ls_sess:
    delta = best_alpha_delta.loc[sess]['delta_punish']
    sensory = encode_sensory(trmtx_sub, delta)
    alpha = best_alpha_delta.loc[sess]['alpha_learning']
    motivational = encode_motivational(trmtx_sub, alpha, negative_scale=negative_scale)

    for loc in ls_loc:
        idx_trial_select = (trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == loc)
        if idx_trial_select.sum() == 0:
            continue
        idx_trial_select = idx_trial_select.to_numpy()

        y = mean_post_reinf[idx_trial_select]
        X = np.array([motivational[idx_trial_select], sensory[idx_trial_select]]).T

        clf = Ridge(alpha=0)
        clf.fit(X, y)

        results_fixed_all.append({
            'an': trmtx_sub.loc[idx_trial_select]['anid'].unique()[0],
            'sess': sess,
            'loc': loc,
            'alpha_learning': alpha,
            'delta_punish': delta,
            'b0': clf.intercept_,
            'b_motiv': clf.coef_[0],
            'b_senso': clf.coef_[1],
            'r2_score': clf.score(X, y),
        })

results_fixed_all = pd.DataFrame(results_fixed_all)

# One row per (sess, loc) — trivial idxmax since each combo has a single fit,
# kept for consistency with the source pipeline.
best_value_fixed = results_fixed_all.loc[results_fixed_all.groupby(['sess', 'loc'])['r2_score'].idxmax()].reset_index(drop=True)
best_value_fixed['loc'] = pd.Categorical(best_value_fixed['loc'], categories=ls_loc, ordered=True)

# Sensory preference index: +1 = purely sensory-driven, -1 = purely
# motivational/reward-value-driven.
best_value_fixed['senso_bias'] = (
    (best_value_fixed['b_senso'].abs() - best_value_fixed['b_motiv'].abs())
    / (best_value_fixed['b_motiv'].abs() + best_value_fixed['b_senso'].abs())
)

best_value_fixed.to_csv('Data/results_multisite_regression2.csv', index=False)


# %% Fig S4 A: anticipatory lick rate vs. fitted alpha

best_alpha_sess = best_value_fixed.groupby('sess')['alpha_learning'].mean()
for_graph = []
for sess in ls_sess:
    idx = (trmtx_sub['sessid'] == sess) & (trmtx_sub['trial_type'] == 'cued rew')
    for_graph.append({
        'sess': sess,
        'alpha': best_alpha_sess[sess],
        'lick': trmtx_sub['Anticipatory(tUS-CS) l/s'].loc[idx].mean(),
    })
for_graph = pd.DataFrame(for_graph)

fig, ax = plt.subplots(figsize=(3, 3))
a = ax
sns.scatterplot(for_graph, x='lick', y='alpha', color='k', ax=a)
r, p = stats.pearsonr(for_graph['lick'], for_graph['alpha'])
a.set_title(f'r={r:0.2f} p={p:0.3f}')
a.set_xlabel('Lick_rate (l/s)')
a.set_ylabel('ɑ (association bias)')
fig.tight_layout()


# %% Fig 3 E, F, G: beta_S, beta_M, sensory preference index by region

fig, ax = plt.subplots(1, 3, figsize=(6, 3))

var = 'b_senso'  # Fig 3E
a = ax[0]
sns.lineplot(best_value_fixed, x='loc', y=var, estimator=None, units='sess',
             ax=a, color='crimson', alpha=0.3, marker='.', size=3, markeredgewidth=0.25)
sns.lineplot(best_value_fixed, x='loc', y=var,
             errorbar='se', err_style='bars', ax=a, color='crimson', marker='o', markeredgewidth=0.25)
a.legend().remove()
a.set_xlabel('')
plt.setp(a.get_xticklabels(), rotation=90, ha='right')
a.set_ylim(0, 4)

var = 'b_motiv'  # Fig 3F
a = ax[1]
sns.lineplot(best_value_fixed, x='loc', y=var, estimator=None, units='sess',
             ax=a, color='k', alpha=0.3, marker='.', size=3, markeredgewidth=0.25)
sns.lineplot(best_value_fixed, x='loc', y=var,
             errorbar='se', err_style='bars', ax=a, color='k', marker='o', markeredgewidth=0.25)
a.legend().remove()
a.set_xlabel('')
plt.setp(a.get_xticklabels(), rotation=90, ha='right')
a.set_ylim(0, 4)

var = 'senso_bias'  # Fig 3G
a = ax[2]
sns.lineplot(best_value_fixed, x='loc', y=var, estimator=None, units='sess',
             ax=a, color='k', alpha=0.3, marker='.', size=3, markeredgewidth=0.25)
sns.lineplot(best_value_fixed, x='loc', y=var,
             errorbar='se', err_style='bars', ax=a, color='k', marker='o', markeredgewidth=0.25)
a.axhline(0, ls='--', color='k')
a.set_ylim(-1, 1)
a.set_xlabel('')
plt.setp(a.get_xticklabels(), rotation=90, ha='right')
a.legend().remove()

fig.tight_layout()


# %% Stats for Fig 3C, 3E, 3F, 3G (repeated-measures ANOVA + Holm post-hocs)

for label, dv_col in [('R2_score', 'r2_score'), ('b_senso', 'b_senso'),
                      ('b_motiv', 'b_motiv'), ('Senso_bias', 'senso_bias')]:
    res = test_rmanova_by_loc(best_value_fixed, id_col="sess", loc_col="loc", dv_col=dv_col, mc_method="holm")
    print()
    print(f'{label}:')
    print(res['anova_table'])
    print(res['posthocs'][['level_a', 'level_b', 'n', 'p_unc', 'p_corr', 'reject']])


# %% Fig 3C, 3D: R2 score, plain and vs. shuffled-regressor control

rng = np.random.default_rng(25)
n_shuffle = 100

results_scrambled = []
for sess in ls_sess:
    for loc in ls_loc:
        idx_trial_select = (trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == loc)
        if idx_trial_select.sum() == 0:
            continue

        alpha, delta, r2_full = best_value_fixed[['alpha_learning', 'delta_punish', 'r2_score']][
            (best_value_fixed['sess'] == sess) & (best_value_fixed['loc'] == loc)
        ].to_numpy()[0]

        idx_trial_select = idx_trial_select.to_numpy()
        trmtx_sub_loc = trmtx_sub.iloc[idx_trial_select]

        sensory = encode_sensory(trmtx_sub_loc, delta)
        motivational = encode_motivational(trmtx_sub_loc, alpha, negative_scale=negative_scale)

        y = mean_post_reinf[idx_trial_select]
        X = np.array([motivational, sensory]).T

        r2_Msh = []
        for _ in range(n_shuffle):
            idx = np.arange(X.shape[0])
            rng.shuffle(idx)
            X_sh = X.copy()
            X_sh[:, 0] = X_sh[idx, 0]
            clf = Ridge(alpha=0)
            clf.fit(X_sh, y)
            r2_Msh.append(clf.score(X, y))

        r2_Ssh = []
        for _ in range(n_shuffle):
            idx = np.arange(X.shape[0])
            rng.shuffle(idx)
            X_sh = X.copy()
            X_sh[:, 1] = X_sh[idx, 1]
            clf = Ridge(alpha=0)
            clf.fit(X_sh, y)
            r2_Ssh.append(clf.score(X, y))

        results_scrambled.append({
            'an': trmtx_sub.loc[idx_trial_select]['anid'].unique()[0],
            'sess': sess,
            'loc': loc,
            'alpha_learning': alpha,
            'delta_punish': delta,
            'r2_full': r2_full,
            'r2_M_shuffled': np.mean(r2_Msh),
            'r2_S_shuffled': np.mean(r2_Ssh),
        })

results_scrambled = pd.DataFrame(results_scrambled)
results_scrambled['loc'] = pd.Categorical(results_scrambled['loc'], categories=ls_loc, ordered=True)

fig, ax = plt.subplots(2, figsize=(2.2, 4))

# Fig 3C: plain R2 by region
a = ax[0]
sns.lineplot(results_scrambled, x='loc', y='r2_full', estimator=None, units='sess',
             ax=a, color='k', alpha=0.3, marker='.', size=3, markeredgewidth=0.25)
sns.lineplot(results_scrambled, x='loc', y='r2_full',
             errorbar='se', err_style='bars', ax=a, color='k', marker='o', markeredgewidth=0.25)
a.legend().remove()
a.set_xlabel('')
a.set_ylim(-0.05, 1.05)
a.set_xlim([-0.5, 3.5])

# Fig 3D: full model vs. shuffled-regressor controls
mtx = results_scrambled[['r2_full', 'r2_M_shuffled', 'r2_S_shuffled']].to_numpy()
ls_marker_color = ['k', 'crimson', 'grey']
for i, loc in enumerate(ls_loc):
    mtx_sub = mtx[(results_scrambled['loc'] == loc).to_numpy()]

    a = ax[1]
    a.plot(np.array([0, 1, 2]) + 3.5 * i, mtx_sub.T, color='k', lw=0.25, alpha=0.5)
    a.set_xticks([1, 4.5, 8, 11.5])
    a.set_xticklabels(ls_loc)

    m = np.mean(mtx_sub, axis=0)
    sem = stats.sem(mtx_sub, axis=0)
    a.plot(np.array([0, 1, 2]) + 3.5 * i, m, color='k', lw=0.75)
    for j in [0, 1, 2]:
        a.errorbar(j + 3.5 * i, m[j], sem[j], marker='o', lw=0.75,
                   color=ls_marker_color[j], markeredgecolor='w', markeredgewidth=0.25)
    a.set_ylim(-0.05, 1.05)

    p1 = stats.ttest_rel(mtx_sub[:, 0], mtx_sub[:, 1])[1] * 2
    p2 = stats.ttest_rel(mtx_sub[:, 0], mtx_sub[:, 2])[1] * 2
    print(f'\n{loc}:\np(M){p1:0.1e}\np(S){p2:0.1e}')

ax[0].set_ylabel('R2 score')
ax[1].set_ylabel('R2 score')
fig.tight_layout()


# %% Fig 3B: predicted GRAB-DA, example session
# NOTE: this loop fits and predicts for every session x region, but only one
# example session gets plotted below (the multi-session panel that used to
# consume the rest is commented out in the original script). If you don't
# need predictions for every session, restrict the outer loop to
# `ls_sess = ['DA4fib02_2024-02-26']` to skip the unused fits.

res_predict = []
for sess in ls_sess:
    for loc in ls_loc:
        idx_trial_select = (trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == loc)
        if idx_trial_select.sum() == 0:
            continue

        alpha, delta, r2_full = best_value_fixed[['alpha_learning', 'delta_punish', 'r2_score']][
            (best_value_fixed['sess'] == sess) & (best_value_fixed['loc'] == loc)
        ].to_numpy()[0]

        idx_trial_select = idx_trial_select.to_numpy()
        trmtx_sub_loc = trmtx_sub.iloc[idx_trial_select]

        sensory = encode_sensory(trmtx_sub_loc, delta)
        motivational = encode_motivational(trmtx_sub_loc, alpha, negative_scale=negative_scale)

        y = mean_post_reinf[idx_trial_select]
        X = np.array([motivational, sensory]).T

        clf = Ridge(alpha=0)
        clf.fit(X, y)
        y_pred = clf.predict(X)

        for i in range(len(trmtx_sub_loc)):
            res_predict.append({
                'sess': trmtx_sub_loc['sessid'].iloc[i],
                'loc': trmtx_sub_loc['loc'].iloc[i],
                'trial_type': trmtx_sub_loc['trial_type'].iloc[i],
                'grab_reinf': trmtx_sub_loc['grab_reinf'].iloc[i],
                'grab_predicted': y_pred[i],
            })

res_predict = pd.DataFrame(res_predict)
res_predict['loc'] = pd.Categorical(res_predict['loc'], categories=ls_loc, ordered=True)
res_predict['trial_type'] = pd.Categorical(res_predict['trial_type'], categories=ls_trial_type, ordered=True)

ls_xticks = ['Uncued rew.', 'Cued rew.', 'Omission', 'Air puff', 'Omiss. & air']
example_sess = 'DA4fib02_2024-02-26'

fig, ax = plt.subplots(figsize=(2, 3))
a = ax
sns.lineplot(res_predict.loc[res_predict['sess'] == example_sess], x='trial_type', y='grab_predicted', hue='loc',
             err_style='bars', marker='o', lw=1, legend=False, errorbar='se', markeredgewidth=0.25, ax=a)
a.axhline(0, ls='--', color='k', alpha=0.5)
a.set_xlabel('')
a.set_ylabel('Predicted GRAB-DA (z-score)')
a.set_title(example_sess, size=6)
a.set_xlim(-0.5, 4.5)
a.set_xticklabels(ls_xticks, rotation=90)
fig.tight_layout()


# %% Fig S4 H (model side only): Pearson r between R and S regressors,
# using each session's fitted alpha/delta applied to one idealized trial of
# each type.

pearsonr_regressors = []
for sess in ls_sess:
    alpha = best_alpha_delta.loc[sess, 'alpha_learning']
    delta = best_alpha_delta.loc[sess, 'delta_punish']

    # One idealized trial of each type: uncued rew, cued rew, omission, uncued pun, omission pun
    X = pd.DataFrame({'ToneID': [3, 2, 2, 3, 2], 'Reward?': [5, 5, 0, -1, -1]})

    R = encode_motivational(X, alpha, negative_scale=negative_scale)
    S = encode_sensory(X, delta)

    pearsonr_regressors.append(stats.pearsonr(R, S)[0])

fig, ax = plt.subplots(figsize=(2, 3))
a = ax
sns.boxplot(pearsonr_regressors, ax=a, fill=False, color='k', width=0.5)
sns.swarmplot(pearsonr_regressors, ax=a, color='k', alpha=0.5)
a.axhline(0, color='k', ls=':', lw=0.5, alpha=0.5)
a.set_ylabel('Pearson r')
a.set_ylim(-1, 0)
fig.tight_layout()

print('mean pearson r (R vs S) = {:0.2f} ± {:0.2f}'.format(
    np.mean(pearsonr_regressors), stats.sem(pearsonr_regressors)))