#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 21:27:06 2026

@author: vbp

Peak-aligned GRAB-DA kinetics for uncued reward, for Figure S2E/F.

    Fig S2E - peak-normalized GRAB-DA traces aligned to each session's own
              response peak, per region
    Fig S2F - full width at half maximum (FWHM) of that peak, per region
              (+ repeated-measures ANOVA across regions)

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

# Crop raster and adjust timing
t_raster += 0.125
crop_win = [-5, 8]
raster_sub = raster_sub[:, (t_raster > crop_win[0]) & (t_raster < crop_win[1])]
t_raster = t_raster[(t_raster > crop_win[0]) & (t_raster < crop_win[1])]

# Some fixing for DA4fib01
trmtx_sub['loc'] = trmtx_sub['loc'].replace({'NAc_c': 'NAc_lat'})
ls_loc = ['DS', 'NAc_lat', 'BLA', 'mPFC']
ls_sess = trmtx_sub['sessid'].unique()

ls_col_loc = ["#ff595e", "#ff924c", "#52a675", "#4267ac"]

TRIAL_SELECTOR = [3, 5]   # ToneID, Reward? for uncued reward
TRIAL_LABEL = 'uncued rew'

# Sample rate implied by the +/-1s window below (80 samples over 4s => 20 Hz)
FS = 20
HALF_WIN_SAMPLES = 20   # samples before the peak
FULL_WIN_SAMPLES = 80   # total window length


# %% Fig S2E, S2F: peak-aligned kinetics and FWHM, per session x region

plt.figure()  # quick diagnostic overlay of every session's aligned trace, not a saved panel

results = []
results_time_serie = []
for sess in ls_sess:
    # Sort/align reference: find each session's DS peak location (not used
    # for sorting here, just to define the trial subset consistently with
    # the source pipeline).
    idx_trial_select = (trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == 'DS')
    idx_trial_select = idx_trial_select & (trmtx_sub['ToneID'] == TRIAL_SELECTOR[0]) & (trmtx_sub['Reward?'] == TRIAL_SELECTOR[1])
    idx_trial_select = idx_trial_select.to_numpy()

    for i, loc in enumerate(ls_loc):
        idx_trial_select = (trmtx_sub['sessid'] == sess) & (trmtx_sub['loc'] == loc)
        if idx_trial_select.sum() == 0:
            continue

        idx_trial_select = idx_trial_select & (trmtx_sub['ToneID'] == TRIAL_SELECTOR[0]) & (trmtx_sub['Reward?'] == TRIAL_SELECTOR[1])
        idx_trial_select = idx_trial_select.to_numpy()

        ras = raster_sub[idx_trial_select]
        if len(ras) == 0:
            continue

        sess_avg = np.mean(ras, axis=0)

       
        peak_idx = np.argmax(sess_avg)
        idx_win = np.arange(FULL_WIN_SAMPLES) - HALF_WIN_SAMPLES + peak_idx

        sess_avg = sess_avg[idx_win]
        sess_avg /= np.max(sess_avg)

        t_win = (np.arange(FULL_WIN_SAMPLES) - HALF_WIN_SAMPLES) / FS

        plt.plot(sess_avg, color=ls_col_loc[i])

        fwhm = np.sum(sess_avg > 0.5) / FS

        results.append({'sess': sess, 'trial_type': TRIAL_LABEL, 'loc': loc, 'fwhm': fwhm})
        for t, s in zip(t_win, sess_avg):
            results_time_serie.append({'sess': sess, 'trial_type': TRIAL_LABEL, 'loc': loc, 't': t, 'mean': s})

results = pd.DataFrame(results)
results_time_serie = pd.DataFrame(results_time_serie)


# %% Fig S2E: peak-normalized traces, Fig S2F: FWHM by region

fig, ax = plt.subplots(1, 2, figsize=(6, 3))

a = ax[0]
sns.lineplot(results_time_serie, x='t', y='mean', hue='loc', errorbar='se', ax=a)
a.set_xlim([-1, 3])
a.set_xlabel('Time from peak\npost-uncued reward (s)')
a.set_ylabel('GRAB-DA (peak norm.)')

a = ax[1]
sns.lineplot(results, x='loc', y='fwhm', estimator=None, units='sess',
             ax=a, color='k', alpha=0.3, linewidth=0.25, marker='o', markersize=2)
sns.lineplot(results, x='loc', y='fwhm', errorbar='se',
             ax=a, color='k', err_style='bars', marker='o', markersize=5)
a.set_ylim(0, 2.6)
a.set_ylabel('FWHM (s)')
a.set_xlabel('')
a.set_xlim(-0.5, 3.5)
plt.setp(a.get_xticklabels(), rotation=90, ha='right')

anova_res = test_rmanova_by_loc(results, id_col='sess', loc_col='loc', dv_col='fwhm')
pval = anova_res['anova_table']['PR(>F)']['C(loc)']
a.set_title(f'{pval:0.3e}')

print()
print(anova_res['posthocs'])

fig.tight_layout()
