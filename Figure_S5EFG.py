#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:21:46 2025

@author: vbp

Model fit for the additional target regions (TS, NAc_c, OT), for Figure
S5E, S5F, S5G.

IMPORTANT: unlike the main-region regression scripts, this one does NOT
fit alpha/delta per session — it uses fixed values (alpha=0.83, delta=0.82)
for every session and region. These values are the population mean from best_value_fixed2.csv data.

    Fig S5E - R2 score across all 7 regions (OT, NAc_c, NAc_lat, DS, BLA, TS, mPFC)
    Fig S5F - R2 score, full model vs. shuffled-regressor controls, for
              OT, NAc_c, TS only
    Fig S5G - sensory preference index across all 7 regions (+ ANOVA)

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
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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


# %% Stats: one-way ANOVA with Tukey HSD post-hocs

def oneway_anova_with_posthocs(df: pd.DataFrame, dv: str = "val", condition: str = "condition",
                                *, alpha: float = 0.05, typ: int = 2):
    """
    One-way ANOVA on dv across levels of 'condition', with Tukey HSD post-hoc
    tests (OLS model: dv ~ C(condition); ANOVA table is Type II by default).

    Returns a dict with anova_table, partial_eta_sq, model, tukey_condition
    (long/tidy Tukey results), p_adj_matrix / reject_matrix (square matrices
    for heatmaps), assumption checks (Shapiro/Jarque-Bera, Levene), and
    group_sizes.
    """
    needed = {dv, condition}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    data = df[[dv, condition]].dropna().copy()
    if data.empty:
        raise ValueError("No data left after dropping NA rows.")

    if not pd.api.types.is_categorical_dtype(data[condition]):
        data[condition] = data[condition].astype("category")

    if data[condition].nunique() < 2:
        raise ValueError("Need at least two levels in 'condition' to run one-way ANOVA.")

    formula = f"{dv} ~ C({condition})"
    model = smf.ols(formula, data=data).fit()
    aov = anova_lm(model, typ=typ)

    cond_row = f"C({condition})"
    if cond_row not in aov.index:
        raise RuntimeError(f"Could not find factor '{cond_row}' in ANOVA table.")
    ss_effect = aov.loc[cond_row, "sum_sq"]
    ss_error = aov.loc["Residual", "sum_sq"]
    partial_eta_sq = float(ss_effect / (ss_effect + ss_error)) if (ss_effect + ss_error) > 0 else np.nan

    try:
        shapiro_W, shapiro_p = stats.shapiro(model.resid)
    except Exception:
        jb_stat, jb_p, _, _ = sm.stats.jarque_bera(model.resid)
        shapiro_W, shapiro_p = np.nan, jb_p

    groups = [data.loc[data[condition] == lvl, dv].values for lvl in data[condition].cat.categories]
    levene_stat, levene_p = stats.levene(*groups, center="median")

    tukey = pairwise_tukeyhsd(endog=data[dv].values, groups=data[condition].values, alpha=alpha)
    tukey_df = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    for col in ["meandiff", "lower", "upper", "p-adj"]:
        tukey_df[col] = pd.to_numeric(tukey_df[col], errors="coerce")
    tukey_df["reject"] = tukey_df["reject"].astype(bool)

    levels = list(data[condition].cat.categories)
    p_mat = pd.DataFrame(np.nan, index=levels, columns=levels, dtype=float)
    r_mat = pd.DataFrame(False, index=levels, columns=levels, dtype=bool)
    np.fill_diagonal(p_mat.values, 0.0)
    np.fill_diagonal(r_mat.values, False)

    for _, row in tukey_df.iterrows():
        g1 = str(row["group1"])
        g2 = str(row["group2"])
        p = float(row["p-adj"])
        rej = bool(row["reject"])
        if g1 in p_mat.index and g2 in p_mat.columns:
            p_mat.loc[g1, g2] = p
            p_mat.loc[g2, g1] = p
            r_mat.loc[g1, g2] = rej
            r_mat.loc[g2, g1] = rej

    return {
        "anova_table": aov,
        "partial_eta_sq": partial_eta_sq,
        "model": model,
        "tukey_condition": tukey_df,
        "p_adj_matrix": p_mat,
        "reject_matrix": r_mat,
        "assumptions": {
            "shapiro_resid": (shapiro_W, shapiro_p),
            "levene_across_conditions": (levene_stat, levene_p),
        },
        "group_sizes": data.groupby(condition, observed=True)[dv].size(),
    }


# %% Load data and select sessions
#
# Two-stage selection, additive (OR'd into the same mask, not reset in
# between): stage 1 keeps multi-site sessions across the 4 main regions;
# stage 2 adds sessions recorded from the 3 additional-target regions.

set_up_figure_format()

DATA_PATH = 'Data/av_and_probcond_raster.npy'
data = np.load(DATA_PATH, allow_pickle=True).item()
trmtx = data['trmtx']
raster_all_data = data['raster']
t_raster = data['t_raster']
ls_sess = trmtx['sessid'].unique()

idx_select = np.zeros((len(trmtx)), dtype=bool)

ls_loc_multisite_filter = ['DS', 'NAc_lat', 'BLA', 'mPFC', 'NAc_c']
for sess in ls_sess:
    sub_loc = trmtx['loc'][trmtx['sessid'] == sess].unique()
    exp = trmtx['experiment'][trmtx['sessid'] == sess].unique()
    if len(sub_loc) > 2:
        if set(sub_loc).issubset(ls_loc_multisite_filter):
            if set(['Av']).issubset(exp):
                idx_select = idx_select | ((trmtx['sessid'] == sess).to_numpy() & (trmtx['experiment'] == 'Av').to_numpy())

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

t_raster += 0.125
crop_win = [-0.5, 3.5]
raster_sub = raster_sub[:, (t_raster > crop_win[0]) & (t_raster < crop_win[1])]
t_raster = t_raster[(t_raster > crop_win[0]) & (t_raster < crop_win[1])]

ls_loc_all = ['DS', 'TS', 'NAc_lat', 'NAc_c', 'OT', 'BLA', 'mPFC']
ls_sess = trmtx_sub['sessid'].unique()


# %% Model parameters

ls_trial = [
    [3, 5],   # Uncued reward
    [2, 5],   # Cued reward
    [2, 0],   # Reward omission
    [3, -1],  # Uncued air puff
    [2, -1],  # Reward omission & air puff
]
ls_trial_type = ['uncued rew', 'cued rew', 'omission', 'uncued pun', 'omission pun']

fit_win = [1.5, 2.5]  # window used to fit the model (relative to cue)
negative_scale = 0.5

mean_post_reinf = np.mean(raster_sub[:, (t_raster > fit_win[0]) & (t_raster < fit_win[1])], axis=1)
bl = np.mean(raster_sub[:, (t_raster > fit_win[0] - 0.25) & (t_raster < fit_win[0])], axis=1)
mean_post_reinf -= bl


# %% Fit with fixed alpha/delta (see note at top of file) for every region

# TODO: confirm 0.83 / 0.82 are the intended values (e.g. population mean
# alpha/delta from the main-region model) and document their provenance.
FIXED_ALPHA = 0.83
FIXED_DELTA = 0.82

results_fixed_all = []
for sess in ls_sess:
    sensory = encode_sensory(trmtx_sub, FIXED_DELTA)
    motivational = encode_motivational(trmtx_sub, FIXED_ALPHA, negative_scale=negative_scale)

    for loc in ls_loc_all:
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
            'alpha_learning': FIXED_ALPHA,
            'delta_punish': FIXED_DELTA,
            'b0': clf.intercept_,
            'b_motiv': clf.coef_[0],
            'b_senso': clf.coef_[1],
            'r2_score': clf.score(X, y),
        })

results_fixed_all = pd.DataFrame(results_fixed_all)

best_value_fixed = results_fixed_all.loc[results_fixed_all.groupby(['sess', 'loc'])['r2_score'].idxmax()].reset_index(drop=True)

ls_loc_ordered = ['OT', 'NAc_c', 'NAc_lat', 'DS', 'BLA', 'TS', 'mPFC']
best_value_fixed['loc'] = pd.Categorical(best_value_fixed['loc'], categories=ls_loc_ordered, ordered=True)
results_fixed_all['loc'] = pd.Categorical(results_fixed_all['loc'], categories=ls_loc_ordered, ordered=True)

best_value_fixed['senso_bias'] = (
    (best_value_fixed['b_senso'].abs() - best_value_fixed['b_motiv'].abs())
    / (best_value_fixed['b_motiv'].abs() + best_value_fixed['b_senso'].abs())
)


# %% Fig S5G: sensory preference index across all 7 regions 

fig, ax = plt.subplots(1, 3, figsize=(6, 3))

var = 'b_senso'
a = ax[0]
sns.lineplot(best_value_fixed, x='loc', y=var, estimator=None, units='sess',
             ax=a, color='crimson', alpha=0.3, marker='.', size=3, markeredgewidth=0.25)
sns.lineplot(best_value_fixed, x='loc', y=var,
             errorbar='se', err_style='bars', ax=a, color='crimson', marker='o', markeredgewidth=0.25)
a.legend().remove()
a.set_xlabel('')
plt.setp(a.get_xticklabels(), rotation=90, ha='right')
a.set_ylim(0, 4)

var = 'b_motiv'
a = ax[1]
sns.lineplot(best_value_fixed, x='loc', y=var, estimator=None, units='sess',
             ax=a, color='k', alpha=0.3, marker='.', size=3, markeredgewidth=0.25)
sns.lineplot(best_value_fixed, x='loc', y=var,
             errorbar='se', err_style='bars', ax=a, color='k', marker='o', markeredgewidth=0.25)
a.legend().remove()
a.set_xlabel('')
plt.setp(a.get_xticklabels(), rotation=90, ha='right')
a.set_ylim(0, 4)

var = 'senso_bias'
a = ax[2]
sns.stripplot(best_value_fixed, x='loc', y=var, ax=a, color='k', alpha=0.3, marker='.', size=5)
sns.lineplot(best_value_fixed, x='loc', y=var, lw=0,
             errorbar='se', err_style='bars', ax=a, color='k', marker='o', markeredgewidth=0.25)
a.axhline(0, ls='--', color='k')
a.set_ylim(-1.05, 1.05)
a.set_xlabel('')
plt.setp(a.get_xticklabels(), rotation=90, ha='right')
a.legend().remove()
fig.tight_layout()

res = oneway_anova_with_posthocs(best_value_fixed, 'senso_bias', 'loc')
print()
print('Senso_bias:')
print(res['anova_table'])
print(res['tukey_condition'])


# %% Fig S5E, S5F: R2 score, plain (all 7 regions) and vs. shuffled-regressor
# controls (OT, NAc_c, TS only)

rng = np.random.default_rng(25)
n_shuffle = 100

results_scrambled = []
for sess in ls_sess:
    for loc in ls_loc_all:
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
results_scrambled['loc'] = pd.Categorical(results_scrambled['loc'], categories=ls_loc_ordered, ordered=True)

fig, ax = plt.subplots(1, 2, figsize=(5, 3))

# Fig S5E: R2 score, all 7 regions
a = ax[0]
sns.stripplot(results_scrambled, x='loc', y='r2_full', ax=a, color='k', alpha=0.3, marker='.', size=6)
sns.lineplot(results_scrambled, x='loc', y='r2_full', lw=0,
             errorbar='se', err_style='bars', ax=a, color='k', marker='o', markeredgewidth=0.25)
a.legend().remove()
a.set_xlabel('')
plt.setp(a.get_xticklabels(), rotation=90, ha='right')
a.set_ylim(-0.05, 1.05)

res = oneway_anova_with_posthocs(results_scrambled, 'r2_full', 'loc')
print('R2 Full:')
print(res['anova_table'])
print(res['tukey_condition'])

# Fig S5F: full model vs. shuffled-regressor controls, OT/NAc_c/TS only
mtx = results_scrambled[['r2_full', 'r2_M_shuffled', 'r2_S_shuffled']].to_numpy()
ls_marker_color = ['k', 'crimson', 'grey']
ls_loc_targets3 = ['OT', 'NAc_c', 'TS']

for i, loc in enumerate(ls_loc_targets3):
    mtx_sub = mtx[(results_scrambled['loc'] == loc).to_numpy()]

    a = ax[1]
    a.plot(np.array([0, 1, 2]) + 3.5 * i, mtx_sub.T, color='k', lw=0.25, alpha=0.5)
    a.set_xticks([1, 4.5, 8])
    a.set_xticklabels(ls_loc_targets3)

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
