#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:31:52 2024

@author: vbp

Figure S2B: anticipatory (CS-, CS+) and consummatory lick rate, per animal.

Data can be downloaded here: https://doi.org/10.17605/OSF.IO/DV724
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats


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


# %% Load lick-rate data

set_up_figure_format()

# THIS CSV IS CREATED FROM 'compile_DataLick.py'
DATA_PATH = 'Data/dataLick.csv'
df_compil = pd.read_csv(DATA_PATH)

# df_compil is part of a larger dataset; keep only the sessions relevant here
ls_an_sel = ['DA4fib01', 'DA4fib02', 'DA4fib03', 'DA4fib04', 'DA4fib06', 'DA4fib07', 'DA4fib08']

ls_ansimple = [an[:an.find('_')] for an in df_compil['anid']]
idx_select = [a in ls_an_sel for a in ls_ansimple]

df_compil['ansimple'] = ls_ansimple
# NOTE: [:, 1:] drops the CSV's first column (assumed to be a saved pandas
# index from compile_DataLick.py, not data) — check this still holds if the
# CSV-writing step ever changes.
df_compil = df_compil.iloc[idx_select, 1:].reset_index(drop=True)
df_compil = df_compil.rename(columns={'cond': 'loc'})

# Some fixing for DA4fib01
df_compil['loc'] = df_compil['loc'].replace({'NAc_c': 'NAc_lat'})


# %% Fig S2B: anticipatory CS-/CS+ and consummatory lick rate, per animal

data_lick = df_compil.groupby('ansimple')[['anti_cs_neg', 'anti_cs_pos', 'rew_exp']].mean().to_numpy()

fig, ax = plt.subplots(figsize=(3, 4))
a = ax
a.plot(data_lick.T, color='k', lw=0.5, alpha=0.25)
m = np.nanmean(data_lick, axis=0)
err = stats.sem(data_lick, axis=0, nan_policy='omit')
a.errorbar(np.arange(data_lick.shape[1]), m, err, color='k', marker='o', markerfacecolor='k')
a.set_xticks(np.arange(data_lick.shape[1]))
a.set_xticklabels(['Anticip. CS-', 'Anticip. CS+', 'Consumatory'])
a.set_ylabel('Lick/s')

# Columns: 0 = CS-, 1 = CS+, 2 = Consumatory
st, p1 = stats.ranksums(data_lick[:, 0], data_lick[:, 1])   # CS- vs CS+
st, p2 = stats.ranksums(data_lick[:, 1], data_lick[:, 2])   # CS+ vs Consumatory
a.set_title('p (cs- vs cs+) = {:1.3f}; p (cs+ vs conso) = {:1.3f}'.format(p1, p2), size=8)
fig.tight_layout()
