import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from cycler import cycler

from scipy.stats import chi2
import numpy as np
from emsel.emsel_util import bh_correct, get_1d_s_data_from_type, get_llg_array, get_llgka_array, full_bh_procedure, classify_full_run

###### MODIFY

EM_dir = "EM"
output_dir = "output"

###### DO NOT MODIFY

plt.rcParams.update({'font.size': 9,
                     'text.usetex': False,
                     'font.family': 'serif',
                     'font.serif': 'cmr10',
                     'mathtext.fontset': 'cm',
                     'axes.unicode_minus': False,
                     'axes.formatter.use_mathtext': True,})
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
coolormap = plt.get_cmap("Dark2")
colorlist = ["#1D6996", *[coolormap(i) for i in [1,0]], colors[3], colors[4], colors[5]]
plt.rcParams["axes.prop_cycle"] = cycler(color=colorlist)

genes = ["asip", "mc1r"]
labels = ["A", "B"]
k_opt = 1
chi2_dist = chi2(1)

onep_types = ["add", "dom", "rec", "het"]


for i, gene in enumerate(genes):
    with open(Path(f"{EM_dir}/horse_data_{gene}_EM.pkl"),"rb") as file:
        hf = pickle.load(file)

    n_ll = hf["neutral_ll"][0]
    for sel_type in onep_types:
        nn_ll = hf[f"{sel_type}_run"]["ll_final"][0]
        llr = 2*(nn_ll-n_ll)
        p_val = -chi2_dist.logsf(llr) / np.log(10)
        print(f"{gene} {sel_type} llr: {llr:.3f} logp-value: {p_val:.3f} p-val {np.power(10, -p_val):.6f}")
    full_ll = hf["full_run"]["ll_final"][0]
    d2_stat = 2*(full_ll-n_ll)


    with open(Path(f"{EM_dir}/horse_data_{gene}_permuted_EM.pkl"), "rb") as file:
        permf = pickle.load(file)

    perm_n_lls = permf["neutral_ll"]
    perm_full_lls = permf["full_run"]["ll_final"]
    perm_d2s = 2*(perm_full_lls-perm_n_lls)
    print(f"{gene} empirical percentile: {(perm_d2s<d2_stat).mean():.3f}")
    fig, axs = plt.subplots(1,1,figsize=(3.1, 3.1), layout="constrained")
    axs.hist(perm_d2s, bins=10)
    axs.axvline(d2_stat, color="red", ls="--")
    axs.set_xlabel(r"$D_2$ statistic")
    axs.set_ylabel("Counts")
    axs.text(-.22, .95, rf"$\bf{{{labels[i]}}}$", fontsize=13, transform=axs.transAxes)
    fig.savefig(f"{output_dir}/{gene}_permuted_distribution.pdf", format="pdf", bbox_inches="tight")

