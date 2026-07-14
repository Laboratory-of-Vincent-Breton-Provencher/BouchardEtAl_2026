#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:21:46 2025

@author: vbp

K-fold cross-validation of the R/S regression model

Approach: for each session, split trials into k folds (stratified by
region, so each fold has enough trials per region for the downstream
per-region fit). For each fold: find the best alpha/delta by grid search
on the training folds only, then evaluate R2 and extract beta weights on
the held-out test fold. Compare the cross-validated R2 and sensory
preference index to the original (non-cross-validated) estimates.

Two figures come out of this:
  - `cross_validation_comparison.pdf`: a 3-panel reviewer-response figure
    (original vs. CV R2, original vs. CV sensory preference index, and a
    per-session-x-region scatter of the two). Diagnostic, not a numbered
    figure panel.
  - Figure S4B: R2 score (5-fold CV) by region, the final section below.

Requires `results_multisite_regression2.csv` (best_value_fixed) already
computed by the non-cross-validated regression script. Figure_3B-H_S4ACDEFH.py

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
from sklearn.model_selection import StratifiedKFold


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


# %% Model parameters, trial types

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

tr_type = np.zeros(len(trmtx_sub), dtype=object)
for j, trial_selector in enumerate(ls_trial):
    tr_type[(trmtx_sub['ToneID'] == trial_selector[0]) & (trmtx_sub['Reward?'] == trial_selector[1])] = ls_trial_type[j]
tr_type[tr_type == 0] = 'NA'
trmtx_sub['trial_type'] = tr_type
trmtx_sub['trial_type'] = pd.Categorical(trmtx_sub['trial_type'], categories=ls_trial_type, ordered=True)
trmtx_sub['loc'] = pd.Categorical(trmtx_sub['loc'], categories=ls_loc, ordered=True)

trmtx_sub['grab_cue'] = np.mean(raster_sub[:, (t_raster > 0) & (t_raster < 1)], axis=1)
trmtx_sub['grab_reinf'] = np.mean(raster_sub[:, (t_raster > 1.5) & (t_raster < 2.5)], axis=1)

# TODO: confirm this path — the sibling regression script that produces this
# CSV has saved it to different locations across versions ('Data/...' in one,
# bare filename in another). Point this at wherever your copy actually is.
best_value_fixed = pd.read_csv('Data/results_multisite_regression2.csv')


# %% K-fold cross-validation
#
# For each session, split trials into 5 folds. Grid-search alpha/delta on
# the training folds only (avoiding the circularity of fitting and
# evaluating on the same data), then evaluate R2 and beta weights on the
# held-out fold.
#
# NOTE: this is the expensive part of the script — for each session, this
# is folds(5) x alpha(40) x delta(41) x regions(4) Ridge fits, i.e.
# several hundred thousand fits total. Reduce ls_alpha/ls_delta resolution
# below (they reuse n_alpha/n_delta from above) if this needs to run faster.

n_folds = 5  # standard choice; with ~5 trial types and ~80 trials per
             # session this gives ~16 trials per fold

results_cv = []

for sess in ls_sess:
    print(f'Cross-validating session: {sess}')

    idx_sess = (trmtx_sub['sessid'] == sess).to_numpy()
    trmtx_sess = trmtx_sub.iloc[idx_sess]
    y_sess = mean_post_reinf[idx_sess]

    # Stratify by region (not trial type) so each fold has enough trials
    # per region for the downstream per-region fit below.
    loc_numeric = trmtx_sess['loc'].cat.codes.to_numpy()

    skf = StratifiedKFold(n_splits=n_folds)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(trmtx_sess)), loc_numeric)):
        trmtx_train = trmtx_sess.iloc[train_idx]
        trmtx_test = trmtx_sess.iloc[test_idx]
        y_train = y_sess[train_idx]
        y_test = y_sess[test_idx]

        # ---- Step 1: find best alpha/delta on TRAINING data only ----
        best_r2_train = -np.inf
        best_alpha_cv = None
        best_delta_cv = None

        for alpha in ls_alpha:
            for delta in ls_delta:
                r2_locs = []
                for loc in ls_loc:
                    idx_loc_train = (trmtx_train['loc'] == loc).to_numpy()
                    if idx_loc_train.sum() < 3:  # need a minimum of trials
                        continue

                    mot_train = encode_motivational(trmtx_train.iloc[idx_loc_train], alpha, negative_scale=negative_scale)
                    sen_train = encode_sensory(trmtx_train.iloc[idx_loc_train], delta)

                    X_train = np.array([mot_train, sen_train]).T
                    yy_train = y_train[idx_loc_train]

                    clf = Ridge(alpha=0)
                    clf.fit(X_train, yy_train)
                    r2_locs.append(clf.score(X_train, yy_train))

                # Joint fit: average R2 across regions (same as the
                # non-cross-validated grid search)
                if len(r2_locs) > 0:
                    mean_r2 = np.mean(r2_locs)
                    if mean_r2 > best_r2_train:
                        best_r2_train = mean_r2
                        best_alpha_cv = alpha
                        best_delta_cv = delta

        # ---- Step 2: evaluate on the HELD-OUT test fold ----
        for loc in ls_loc:
            idx_loc_test = (trmtx_test['loc'] == loc).to_numpy()
            idx_loc_train = (trmtx_train['loc'] == loc).to_numpy()
            if idx_loc_test.sum() < 2 or idx_loc_train.sum() < 3:
                continue

            mot_train = encode_motivational(trmtx_train.iloc[idx_loc_train], best_alpha_cv, negative_scale=negative_scale)
            sen_train = encode_sensory(trmtx_train.iloc[idx_loc_train], best_delta_cv)
            X_train = np.array([mot_train, sen_train]).T
            yy_train = y_train[idx_loc_train]

            clf = Ridge(alpha=0)
            clf.fit(X_train, yy_train)

            mot_test = encode_motivational(trmtx_test.iloc[idx_loc_test], best_alpha_cv, negative_scale=negative_scale)
            sen_test = encode_sensory(trmtx_test.iloc[idx_loc_test], best_delta_cv)
            X_test = np.array([mot_test, sen_test]).T
            yy_test = y_test[idx_loc_test]

            r2_test = clf.score(X_test, yy_test)
            print(f'{loc} r2={r2_test:0.3f}')

            b_motiv_cv = clf.coef_[0]
            b_senso_cv = clf.coef_[1]
            senso_bias_cv = (abs(b_senso_cv) - abs(b_motiv_cv)) / (abs(b_senso_cv) + abs(b_motiv_cv))

            results_cv.append({
                'sess': sess,
                'fold': fold_idx,
                'loc': loc,
                'alpha_cv': best_alpha_cv,
                'delta_cv': best_delta_cv,
                'r2_test': r2_test,  # KEY: R2 on held-out data
                'b_motiv_cv': b_motiv_cv,
                'b_senso_cv': b_senso_cv,
                'senso_bias_cv': senso_bias_cv,
            })

results_cv = pd.DataFrame(results_cv)
results_cv['loc'] = pd.Categorical(results_cv['loc'], categories=ls_loc, ordered=True)


# %% Aggregate cross-validated results (mean across folds per session x region)

results_cv_avg = results_cv.groupby(['sess', 'loc']).agg(
    r2_test_mean=('r2_test', 'mean'),
    b_motiv_cv_mean=('b_motiv_cv', 'mean'),
    b_senso_cv_mean=('b_senso_cv', 'mean'),
    senso_bias_cv_mean=('senso_bias_cv', 'mean'),
    alpha_cv_mean=('alpha_cv', 'mean'),
    delta_cv_mean=('delta_cv', 'mean'),
).reset_index()
results_cv_avg['loc'] = pd.Categorical(results_cv_avg['loc'], categories=ls_loc, ordered=True)


# %% Reviewer-response figure: original vs. cross-validated (not a numbered
# figure panel — a diagnostic comparison saved separately)

fig, ax = plt.subplots(1, 3, figsize=(8, 3))

# Panel 1: R2 score, original vs. cross-validated
a = ax[0]
r2_orig = best_value_fixed.groupby(['sess', 'loc'])['r2_score'].mean().reset_index()
r2_orig['loc'] = pd.Categorical(r2_orig['loc'], categories=ls_loc, ordered=True)

sns.lineplot(r2_orig, x='loc', y='r2_score',
             errorbar='se', err_style='bars', ax=a,
             color='k', marker='o', markeredgewidth=0.25, label='Original')
sns.lineplot(results_cv_avg, x='loc', y='r2_test_mean',
             errorbar='se', err_style='bars', ax=a,
             color='steelblue', marker='o', markeredgewidth=0.25,
             label='Cross-validated', linestyle='--')
a.set_ylabel('R² score')
a.set_xlabel('')
a.set_ylim(-0.1, 1.05)
a.set_title('Model fit (R²)')
plt.setp(a.get_xticklabels(), rotation=90, ha='right')
a.legend(fontsize=6)

# Panel 2: Sensory preference index, original vs. cross-validated
a = ax[1]
sns.lineplot(best_value_fixed, x='loc', y='senso_bias',
             errorbar='se', err_style='bars', ax=a,
             color='k', marker='o', markeredgewidth=0.25, label='Original')
sns.lineplot(results_cv_avg, x='loc', y='senso_bias_cv_mean',
             errorbar='se', err_style='bars', ax=a,
             color='steelblue', marker='o', markeredgewidth=0.25,
             label='Cross-validated', linestyle='--')
a.axhline(0, ls='--', color='k', lw=0.5, alpha=0.5)
a.set_ylim(-1, 1)
a.set_ylabel('Sensory preference index')
a.set_xlabel('')
a.set_title('Sensory preference index')
plt.setp(a.get_xticklabels(), rotation=90, ha='right')
a.legend(fontsize=6)

# Panel 3: per-session x region scatter, original vs. CV sensory preference index
a = ax[2]
merged = best_value_fixed[['sess', 'loc', 'senso_bias']].merge(
    results_cv_avg[['sess', 'loc', 'senso_bias_cv_mean']], on=['sess', 'loc'])

colors = ['#ff595e', '#ff924c', '#52a675', '#4267ac']
for i, loc in enumerate(ls_loc):
    sub = merged[merged['loc'] == loc]
    a.scatter(sub['senso_bias'], sub['senso_bias_cv_mean'], color=colors[i], label=loc, s=20, alpha=0.7)

lims = [-1, 1]
a.plot(lims, lims, 'k--', lw=0.5, alpha=0.5, label='Identity')
a.set_xlabel('Original sensory preference index')
a.set_ylabel('Cross-validated sensory preference index')
a.set_xlim(-1, 1)
a.set_ylim(-1, 1)

r, p = stats.pearsonr(merged['senso_bias'], merged['senso_bias_cv_mean'])
a.set_title(f'r={r:.2f}, p={p:.2e}')
a.legend(fontsize=6)

fig.tight_layout()
plt.savefig('cross_validation_comparison.pdf', bbox_inches='tight')


# %% Stats: regional differences in cross-validated metrics

print("\n=== Cross-validated sensory preference index ===")
res_cv_stats = test_rmanova_by_loc(results_cv_avg, id_col="sess", loc_col="loc", dv_col="senso_bias_cv_mean", mc_method="holm")
print(res_cv_stats['anova_table'])
print(res_cv_stats['posthocs'][['level_a', 'level_b', 'n', 'p_unc', 'p_corr', 'reject']])

print("\n=== Cross-validated R2 scores ===")
res_cv_r2_stats = test_rmanova_by_loc(results_cv_avg, id_col="sess", loc_col="loc", dv_col="r2_test_mean", mc_method="holm")
print(res_cv_r2_stats['anova_table'])

print("\n=== Summary ===")
print("Original model R2:")
for loc in ls_loc:
    vals = best_value_fixed[best_value_fixed['loc'] == loc]['r2_score']
    print(f"  {loc}: {vals.mean():.2f} ± {vals.sem():.2f}")

print("\nCross-validated R2 (held-out test folds):")
for loc in ls_loc:
    vals = results_cv_avg[results_cv_avg['loc'] == loc]['r2_test_mean']
    print(f"  {loc}: {vals.mean():.2f} ± {vals.sem():.2f}")

print("\nOriginal sensory preference index:")
for loc in ls_loc:
    vals = best_value_fixed[best_value_fixed['loc'] == loc]['senso_bias']
    print(f"  {loc}: {vals.mean():.2f} ± {vals.sem():.2f}")

print("\nCross-validated sensory preference index:")
for loc in ls_loc:
    vals = results_cv_avg[results_cv_avg['loc'] == loc]['senso_bias_cv_mean']
    print(f"  {loc}: {vals.mean():.2f} ± {vals.sem():.2f}")


# %% Figure S4B: R2 score (5-fold CV) by region

excluded_sess = 'DA4fib06_2024-05-14'
plot_data = results_cv_avg[results_cv_avg['sess'] != excluded_sess]

fig, ax = plt.subplots(figsize=(2.2, 3))
a = ax
sns.lineplot(plot_data, x='loc', y='r2_test_mean', estimator=None, units='sess',
             ax=a, color='k', alpha=0.3, marker='.', size=3, markeredgewidth=0.25, legend=False)
sns.lineplot(plot_data, x='loc', y='r2_test_mean',
             errorbar='se', err_style='bars', ax=a,
             color='k', marker='o', markeredgewidth=0.25, linestyle='--', legend=False)
a.set_ylabel('R² score (5-fold CV)')
a.set_xlabel('')
a.set_ylim(0, 0.9)
fig.tight_layout()
