import numpy as np
from pathlib import Path
import pickle
from scipy.stats import chi2
from scipy.interpolate import CubicSpline
from emsel.emsel_util import bh_correct, get_1d_s_data_from_type
from tqdm import tqdm
import matplotlib.pyplot as plt
from cycler import cycler

###### MODIFY

data_dir = "data"
EM_dir = "EM/old/ne"
output_dir = "output/real_matched"
genodata_type = "capture_only"

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
colorlist = ["#1D6996", *[coolormap(i) for i in [1,0]], colors[3], colors[4]]
init_colorlist = colorlist
plt.rcParams["axes.prop_cycle"] = cycler(color=colorlist)

hmm_Nes = np.geomspace(500,500000, 21, dtype=int)
#sim_Nes = [2500,10000,40000]
prefixes = ["g125_dal_special"]
midfix = ""

sim_Ne = 40000
for prefix in prefixes:
    #for sim_Ne in sim_Nes:
        unif_Ne_lls = np.zeros_like(hmm_Nes, dtype=float)
        datameanic_Ne_lls = np.zeros_like(hmm_Nes, dtype=float)
        icest_Ne_lls = np.zeros_like(hmm_Nes, dtype=float)
        condo_Ne_lls = np.zeros_like(hmm_Nes, dtype=float)
        for Ne_i, hmm_Ne in enumerate(hmm_Nes):
            unif_em_path = Path(f"{EM_dir}/neutral_{prefix}_{midfix}Ne{sim_Ne}_HMMNe{hmm_Ne}_unif_EM.pkl")
            with open(unif_em_path, "rb") as file:
                unif_hf = pickle.load(file)
            unif_Ne_lls[Ne_i] += unif_hf["neutral_ll"].sum()
            datameanic_em_path = Path(f"{EM_dir}/neutral_{prefix}_{midfix}Ne{sim_Ne}_HMMNe{hmm_Ne}_{'delta' if 'd25' in prefix else 'datameanic'}_EM.pkl")
            with open(datameanic_em_path, "rb") as file:
                datameanic_hf = pickle.load(file)
            datameanic_Ne_lls[Ne_i] += datameanic_hf["neutral_ll"].sum()
            icest_em_path = Path(f"{EM_dir}/neutral_{prefix}_{midfix}Ne{sim_Ne}_HMMNe{hmm_Ne}_prof_EM.pkl")
            with open(icest_em_path, "rb") as file:
                icest_hf = pickle.load(file)
            icest_Ne_lls[Ne_i] += icest_hf["neutral_ll"].sum()
            condo_em_path = Path(f"{EM_dir}/neutral_{prefix}_{midfix}Ne{sim_Ne}_HMMNe{hmm_Ne}_cond_unif_EM.pkl")
            with open(condo_em_path, "rb") as file:
                condo_hf = pickle.load(file)
            uncondo_sum = condo_hf["neutral_ll"].sum()
            condo_sum = uncondo_sum - condo_hf["cond_correction_ll"].sum()
            condo_Ne_lls[Ne_i] += condo_sum

            # if Ne_i == 0 and chrom == 1:
            #     print(hf["neutral_itercount"])
            # Ne_lls[Ne_i] += hf["neutral_ll"].sum()

        Nes_space = np.geomspace(500, 500000, 5000)
        icest_condo_spline = CubicSpline(hmm_Nes, icest_Ne_lls)
        icest_y = icest_condo_spline(Nes_space)
        unif_spline = CubicSpline(hmm_Nes, unif_Ne_lls)
        unif_y = unif_spline(Nes_space)
        condo_spline = CubicSpline(hmm_Nes, condo_Ne_lls)
        condo_y = condo_spline(Nes_space)

        icest_argmax = np.argmax(icest_y)
        unif_argmax = np.argmax(unif_y)
        condo_argmax = np.argmax(condo_y)
        print(f"{midfix} est maxes: unif {hmm_Nes[np.argmax(unif_Ne_lls)]}\n icest {hmm_Nes[np.argmax(icest_Ne_lls)]}\n datameanic {hmm_Nes[np.argmax(datameanic_Ne_lls)]}")
        fig, axs = plt.subplots(1,1,figsize=(3.1,3.1),layout="constrained")
        axs.set_xscale("log")
        axs.axvline(sim_Ne, color="k", ls="--")
        axs.plot(hmm_Nes, icest_Ne_lls, ".", label="Est. init. cond.")
        axs.plot(hmm_Nes, unif_Ne_lls, ".", label="Uniform init. cond.")
        axs.plot(hmm_Nes, condo_Ne_lls, ".", label="Conditioned")
        #axs.plot(hmm_Nes, datameanic_Ne_lls, "o", label="data-mean ic")
        axs.set_prop_cycle(None)
        axs.plot(Nes_space, icest_y)
        axs.plot(Nes_space, unif_y)
        axs.plot(Nes_space, condo_y)

        axs.set_prop_cycle(None)
        axs.plot(Nes_space[icest_argmax], icest_y[icest_argmax], "*", ms=10)
        axs.plot(Nes_space[unif_argmax], unif_y[unif_argmax], "*", ms=10)
        axs.plot(Nes_space[condo_argmax], condo_y[condo_argmax], "*", ms=10)
        #axs.plot(hmm_Nes[np.argmax(icest_Ne_lls)], np.max(icest_Ne_lls), "*", ms=20)
        #axs.plot(hmm_Nes[np.argmax(unif_Ne_lls)], np.max(unif_Ne_lls), "*", ms=20)

        #axs.plot(hmm_Nes[np.argmax(datameanic_Ne_lls)], np.max(datameanic_Ne_lls), "*", ms=20)
        axs.legend()
        axs.set_yticks([])
        axs.set_yticklabels([])
        axs.set_xlabel(r"$N_e$")
        axs.set_ylabel("log likelihood (a.u.)")
        #axs.set_title(f"{prefix} {midfix} Ne {sim_Ne}")
        fig.savefig(Path(output_dir)/f"{prefix}_{midfix}Ne{sim_Ne}_40kcondo.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)

#fig, axs = plt.subplots(1,1,figsize=(5,5),layout="constrained")
#axs.boxplot(Ne_iters,tick_labels=Nes)
#fig.savefig(Path(output_dir)/"Nes_itercounts.pdf", format="pdf", bbox_inches="tight")
#plt.close(fig)