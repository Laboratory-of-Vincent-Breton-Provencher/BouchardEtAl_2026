#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:01:30 2026

@author: vbp
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:21:46 2025

@author: vbp

Figure S4I only: sensory preference index (senso_bias) by region. RPE value for punishment set to zero

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


# %% Model encodings

def encode_motivational(trmtx, alpha, negative_scale=1, punish_value=0):
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


# %% Grid search over alpha (association bias) and delta (salience bias)
# to find, for each session x region, the (alpha, delta) that best predicts
# post-reinforcement GRAB-DA from the R (motivational) and S (sensory)
# regressors.

fit_win = [1.5, 2.5]   # window used to fit the model (relative to cue)
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
            # print(f'{sess}: alpha={alpha} delta={delta}')

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

# Best (alpha, delta) per session x region, by R2
best_value = results.loc[results.groupby(['sess', 'loc'])['r2_score'].idxmax()].reset_index(drop=True)
best_value['loc'] = pd.Categorical(best_value['loc'], categories=ls_loc, ordered=True)


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

best_value_fixed.to_csv('Data/results_panel_S4I.csv', index=False)


# %% Figure S4I: sensory preference index by region

fig, ax = plt.subplots(figsize=(2, 3))
a = ax
sns.lineplot(best_value_fixed, x='loc', y='senso_bias', estimator=None, units='sess',
             ax=a, color='k', alpha=0.3, marker='.', size=3, markeredgewidth=0.25)
sns.lineplot(best_value_fixed, x='loc', y='senso_bias',
             errorbar='se', err_style='bars', ax=a, color='k', marker='o', markeredgewidth=0.25)
a.axhline(0, ls='--', color='k')
a.set_ylim(-1, 1)
a.set_xlabel('')
a.set_ylabel('Sensory preference index')
plt.setp(a.get_xticklabels(), rotation=90, ha='right')
a.legend().remove()
fig.tight_layout()

# Stats: repeated-measures ANOVA across regions + Holm-corrected post-hocs
# (these p-values/significant pairs are what the asterisks in Fig S4I show)
res = test_rmanova_by_loc(best_value_fixed, id_col="sess", loc_col="loc", dv_col="senso_bias", mc_method="holm")
print()
print('Sensory preference index (senso_bias):')
print(res['anova_table'])
print(res['posthocs'][['level_a', 'level_b', 'n', 'p_unc', 'p_corr', 'reject']])
