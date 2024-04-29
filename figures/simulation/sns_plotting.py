import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from itertools import product as itprod
from pathlib import Path
import os
import glob
import seaborn as sns
import pandas as pd
from cycler import cycler
import bz2
from scipy.stats import chi2
from util import calculate_converge_idx, plot_one_ll_grid, params_dict_to_str, get_1d_s_data_from_type, convert_from_abbrevs

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

max_axis = 3

subdir_name = "tree/sim"
sel_types = ["neutral", "add", "dom", "rec", "over", "under"]
sel_strs = [.005, .01, .025, .05]
num_gens_list = [101, 251, 1001]
init_dists = [.005, .25, "recip"]

# subdir_name = "tree/real/fake"
# num_gens_list = [125]
# init_dists = ["real_special"]
#init_dists = [.25, .005, "real_special"]

#sel_strs = [.05]
#sel_types = ["add"]

cond_only = False

fake_sel_types = ["neutral"]
#usable_sel_types = ["neutral", "add", "over", "under"]
usable_types = ["neutral","add", "dom", "rec", "over", "under"]
oned_types = ["add", "dom", "rec", "over", "under"]
run_types = ["add", "dom", "rec", "het"]
cond_types = ["neutral", "add", "dom", "rec"]

#sel_types = ["over"]
#sel_strs = [.05]

#run_types = ["add"]
iter_types = cond_types if cond_only else sel_types

thresh = 5

small_s_condos = [False, True]#, True, True]
normal_timesteps = [0, 0]#, 1, .1]

extra_strs = ["-ns 100", "-ns 500", "-ns 500 --hidden_interp chebyshev", "-ns 500 --discrete_M", "-ns 250 --discrete_M"]
file_strs = ["linear_100_noic", "linear_noic", "cheby_noic", "nbeta_final"]
file_strs = ["Ne_10k", "Ne_20k", "Ne_40k"]
init_dist = .25
num_gens = 251

type_strs = ["numerical_beta_partialitertest"]


hmm_prefix_str = "" if "sim" in subdir_name else "tcec_superfinal/"
hmm_fname_str = "linapprox_nbeta_final_" if "sim" in subdir_name else ""
data_prefix_str = "" if "sim" in subdir_name else "finalfinal/"


#for file_i, file_str in enumerate(file_strs):
for n_i, num_gens in enumerate(num_gens_list):
     for d_i, init_dist in enumerate(init_dists):
#         if init_dist in [.005, "recip"] and not cond_only:
#             colorlist = ["#1D6996", coolormap(1), colors[3], colors[4]]
#             plt.rcParams["axes.prop_cycle"] = cycler(color=colorlist)
#         else:
#             colorlist = init_colorlist
#             plt.rcParams["axes.prop_cycle"] = cycler(color=init_colorlist)
        fig, axs = plt.subplots(1, 1, figsize=(3.1, 3.1), layout="constrained")
#     for sq_i, sq_tf in enumerate([False, True]):
        if init_dist == "recip":
            ic_str = "1/x"
        elif init_dist == "real_special":
            ic_str = "real"
        else:
            ic_str = f"delta (p={init_dist})"
        #ic_str = "1/x" if init_dist == "recip" else f"delta (p={init_dist})"
        neutral_list = []
        s_vals = []
        s_types = []
        s_strs = []
        init_conds = []
        freqs_list = []
        min_quantile = 0
        max_quantile = 0
        runtime_total = 0
        illegal_s = 0
        for sel_type, sel_str in itprod(iter_types, sel_strs):
            if init_dist in [.005, "recip"] and sel_type == "rec":
                continue
            pdict = {"sel_type": sel_type, "num_gens": num_gens, "sel_str": sel_str, "init_dist": init_dist}
            exp_name = params_dict_to_str(**pdict)

            if sel_type == "neutral":
                if exp_name not in neutral_list:
                    neutral_list.append(exp_name)
                    sel_str = 0.0
                else:
                    continue

            #remake all these conditional strings so they're not disgusting to look at
            hmm_filename = Path(f"../runs/{subdir_name}/EM/{hmm_prefix_str}{exp_name}_linapprox_{file_str}_EM.bz2")
            #hmm_filename = Path(f"../runs/{subdir_name}/EM/{hmm_prefix_str}{exp_name}_{hmm_fname_str}EM.bz2")
            print(hmm_filename)

            pd_filename = Path(f"../runs/{subdir_name}/datasets/{data_prefix_str}{exp_name}_data.csv")

            pdata_filename = Path(f"../runs/{subdir_name}/datasets/{data_prefix_str}{exp_name}_pd.bz2")

            bh_filename = Path(f"../runs/{subdir_name}/BH/finalfinal/{exp_name}_classified.bz2")

            if not pd_filename.is_file():
                print(f"pd file not found: {pd_filename}")
                continue
            if not hmm_filename.is_file():
                print(f"hmm file not found: {hmm_filename}")
                continue
            if (cond_only and not bh_filename.is_file()):
                print(f"bh file not found: {bh_filename}")
                continue


            with bz2.BZ2File(hmm_filename, "rb") as file:
                hf = pickle.load(file)

            pf = np.loadtxt(pd_filename, delimiter="\t")

            with bz2.BZ2File(pdata_filename, "rb") as file:
                pdict = pickle.load(file)

            num_pts = hf["neutral_ll"].shape[0]
            if cond_only:
                with bz2.BZ2File(bh_filename, "rb") as file:
                    bf = pickle.load(file)
                if "real" in subdir_name:
                    idx_list = np.where(bf["bh_classes"][:500] == sel_types.index(sel_type))[0]
                    num_pts = idx_list.shape[0]
                else:
                    idx_list = np.where(bf["bh_classes"] == sel_types.index(sel_type))[0]
                    num_pts = idx_list.shape[0]
            elif "real" in subdir_name:
                idx_list = np.arange(500)
                num_pts = 500
            else:
                idx_list = np.arange(num_pts)

            if sel_type == "neutral":
                for run_type in run_types:
                    if run_type != "het":
                        run_type_val = run_type
                    else:
                        run_type_val = "over"

                    if cond_only:
                        if run_type in ["add", "dom", "rec"]:
                            idx_list = np.where(bf["bh_classes"] == sel_types.index(run_type))[0]
                            num_pts = idx_list.shape[0]
                            if run_type == "add" and num_pts == 0:
                                s_vals.extend([10])
                                s_types.extend(["add"])
                                s_strs.extend(["Neutral"])
                                subtract_one_from_first_counts = True
                        else:
                            continue
                    if run_type == "rec" and init_dist in [.005, "recip"]:
                        continue
                    exit_codes = hf[f"{run_type}_run"]["exit_codes"]
                    illegal_s += exit_codes[exit_codes == 2].sum()
                    illegal_s += exit_codes[exit_codes == 4].sum()
                    illegal_s += exit_codes[exit_codes == 12].sum()
                    s_data = get_1d_s_data_from_type(hf[f"{run_type}_run"]["s_final"][:, idx_list[:num_pts]], run_type)
                    if s_data.shape[0] > 0:
                        max_quantile = max(max_quantile, np.quantile(s_data, .99))
                        min_quantile = min(min_quantile, np.quantile(s_data, .01))
                    s_vals.extend(s_data.tolist())
                    #s_types.extend([run_type_val] * hf[f"add_run"]["ll_final"][idx_list].shape[0])
                    s_types.extend([f"{run_type_val}"] * num_pts)#idx_list.shape[0])
                    s_strs.extend(["Neutral"] * num_pts)#idx_list.shape[0])
            else:
                if sel_type == "under":
                    exit_codes = hf[f"het_run"]["exit_codes"]
                    illegal_s += exit_codes[exit_codes == 2].sum()
                    illegal_s += exit_codes[exit_codes == 4].sum()
                    illegal_s += exit_codes[exit_codes == 12].sum()
                    s_data = get_1d_s_data_from_type(-hf[f"het_run"]["s_final"][:, idx_list[:num_pts]], sel_type)
                    if s_data.shape[0] > 0:
                        max_quantile = max(max_quantile, np.quantile(s_data, .99))
                        min_quantile = min(min_quantile, np.quantile(s_data, .01))
                    s_vals.extend(s_data.tolist())
                elif sel_type == "over":
                    exit_codes = hf[f"het_run"]["exit_codes"]
                    illegal_s += exit_codes[exit_codes == 2].sum()
                    illegal_s += exit_codes[exit_codes == 4].sum()
                    illegal_s += exit_codes[exit_codes == 12].sum()
                    s_data = get_1d_s_data_from_type(hf[f"het_run"]["s_final"][:, idx_list[:num_pts]], sel_type)
                    if s_data.shape[0] > 0:
                        max_quantile = max(max_quantile, np.quantile(s_data, .99))
                        min_quantile = min(min_quantile, np.quantile(s_data, .01))
                    s_vals.extend(s_data.tolist())
                else:
                    exit_codes = hf[f"{sel_type}_run"]["exit_codes"]
                    illegal_s += exit_codes[exit_codes == 2].sum()
                    illegal_s += exit_codes[exit_codes == 4].sum()
                    illegal_s += exit_codes[exit_codes == 12].sum()
                    s_data = get_1d_s_data_from_type(hf[f"{sel_type}_run"]["s_final"][:, idx_list[:num_pts]], sel_type)
                    if s_data.shape[0] > 0:
                        max_quantile = max(max_quantile, np.quantile(s_data, .99))
                        min_quantile = min(min_quantile, np.quantile(s_data, .01))
                    s_vals.extend(s_data.tolist())
                #s_types.extend([sel_type] * hf[f"add_run"]["ll_final"][idx_list].shape[0])
                s_types.extend([f"{sel_type}"] * num_pts)#idx_list.shape[0])
                s_strs.extend([sel_str] * num_pts)#idx_list.shape[0])

        print(f"illegal s: {illegal_s}")

        s_types = convert_from_abbrevs(s_types, shortall=True)

        massaged_data = zip(s_vals, s_strs, s_types)#

        s_stuff = pd.DataFrame(data=massaged_data, columns=[r"$\hat{s}$", r"True $s$", "Mode of selection"])
        if cond_only:
            box = sns.stripplot(data=s_stuff, x=r"True $s$", y=r"$\hat{s}$", hue="Mode of selection", dodge=True,ax=axs, size=2)
            counts = []
            y_locs = []
            x_locs = []
            for s_i, strength in enumerate(["Neutral", *sel_strs]):
                for m_i, mode in enumerate(["Add.", "Dom.", "Rec."]):
                    bin_mask = ((s_stuff[r"True $s$"]==strength)&(s_stuff["Mode of selection"]==mode)&(s_stuff[r"$\hat{s}$"]<10))
                    counts.append(bin_mask.sum())
                    if counts[-1] > 0:
                        y_locs.append(s_stuff[bin_mask][r"$\hat{s}$"].max())
                    else:
                        y_locs.append(-1)
                    x_locs.append(s_i+(m_i-1)*.28)
                    if counts[-1] > 99:
                        x_locs[-1] += (m_i-1)*.1
            y_locs[0] = y_locs[1]
            for y_i in range(len(y_locs)):
                if y_i%3==2:
                    max_yi = min(max(y_locs[y_i], y_locs[y_i-1], y_locs[y_i-2]), max_quantile)
                    y_locs[y_i] = y_locs[y_i-1] = y_locs[y_i-2] = max_yi*1.05
            for c_i, count in enumerate(counts):
                axs.text(x_locs[c_i], y_locs[c_i], str(count), ha="center", va="bottom", color=colorlist[c_i%3], size=7)
        else:
            box = sns.boxplot(data=s_stuff, x=r"True $s$", y=r"$\hat{s}$", hue="Mode of selection", dodge=True, width=.75,
                  ax=axs,fliersize=1, boxprops={"lw": .5}, medianprops={"lw": .5}, whiskerprops={"lw": .5},
                  capprops={"lw": .5}, flierprops={"alpha": .7}, whis=(2.5, 97.5))
            counts = []
            x_coords = []
            for s_i, strength in enumerate(["Neutral", *sel_strs]):
                for m_i, mode in enumerate(["Add.", "Dom.", "Rec.", "Over.", "Under."]):
                    bin_mask = ((s_stuff[r"True $s$"]==strength)&(s_stuff["Mode of selection"]==mode)&((s_stuff[r"$\hat{s}$"]>max_quantile)|(s_stuff[r"$\hat{s}$"]<min_quantile)))
                    if (bin_counts := bin_mask.sum()) > 0:
                        counts.append(bin_counts)
                    else:
                        counts.append("")
                    x_coords.append(s_i+(m_i-2)*.15)
            for c_i, count in enumerate(counts):
                axs.text(x_coords[c_i], max_quantile*1.01, str(count), ha="center", va="bottom", color=colorlist[c_i%len(colorlist)], size=7)
        axs.axhline(0, color="r", alpha=.65, ls="--", lw=.5)
        for sel_str in sel_strs:
            axs.axhline(sel_str, color="r", alpha=.6, ls="--", lw=.5)
        legend_loc = "lower right" if cond_only else "upper left"
        if not cond_only:
            axs.text(-.24, 1.01, rf"$\bf{{{chr(ord('A')+file_i)}}}$", fontsize=13, transform=axs.transAxes)
        # rf"$\bf{{{chr(ord('A')+file_i)}}}$" OR r"$\bf{A}$"
        #axs.text(.02, .93, rf"$\bf{{{chr(ord('A')+file_i)}}}$", fontsize=50, transform=axs.transAxes)
        axs.legend(loc=legend_loc)#, fontsize=7, labelspacing=.2, handlelength=1.5, handleheight=.5, handletextpad=.4, borderpad=.2, borderaxespad=.2, markerscale=.25 if cond_only else 1)
        # if cond_only:
        #     for handle in axs.get_legend_handles_labels()[0]:
        #         handle.set(alpha=.1)
        #         print(handle.get_alpha())
        #     for handle in axs.get_legend_handles_labels()[0]:
        #         print(handle.get_alpha())
        axs.set_ylim([min_quantile, max_quantile])
        #plt.savefig(f"../runs/{subdir_name}/paper_figures/finalfinal/{'rf_' if 'real' in subdir_name else ''}{'cond' if cond_only else 'all'}_g{num_gens}_d{init_dist}_{'strip' if cond_only else 'box'}plots.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"../runs/{subdir_name}/paper_figures/finalfinal/{file_str}_boxplots.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)

# err_filename = Path(f"../runs/{subdir_name}/EM/linapprox/err/{exp_name}_linapprox_{file_strs[extra_i]}{sq_str}.err")
#
# with open(err_filename, "r") as file:
#     endlist = [line for line in file if "500/500" in line]
#     timelist_0 = [line.split("[")[1] for line in endlist]
#     timelist_1 = [line.split("<")[0] for line in timelist_0]
#
# for timestr in timelist_1:
#     if timestr.count(":") == 0 or timestr.count(":") > 2:
#         raise ValueError
#     if timestr.count(":") == 1:
#         mins, secs = timestr.split(":")
#         runtime_total += 60*int(mins)+int(secs)
#     elif timestr.count(":") == 2:
#         hrs, mins, secs = timestr.split(":")
#         runtime_total += 3600 * hrs + 60 * int(mins) + int(secs)