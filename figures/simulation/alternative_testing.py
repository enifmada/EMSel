import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import numpy as np
from pathlib import Path
import pickle
from bz2 import BZ2File
from util import average_p_vals, params_dict_to_str, bh_correct, get_roc_and_auc, convert_from_abbrevs, plot_qq
from pandas import read_csv
from copy import deepcopy
from scipy.stats import norm, chi2, gamma, gengamma
from pandas import DataFrame
from itertools import product as itprod
import seaborn as sns
import pandas as pd
from cycler import cycler

plt.rcParams.update({'font.size': 9,
                     'text.usetex': False,
                     'font.family': 'serif',
                     'font.serif': 'cmr10',
                     'mathtext.fontset': 'cm',
                     'axes.unicode_minus': False,
                     'axes.formatter.use_mathtext': True,})
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
def cmap(val):
    return (1-val, 0, val, 1)
coolormap = plt.get_cmap("Dark2")
colorlist = ["#1D6996", *[coolormap(i) for i in [1,0]], colors[3], colors[4], colors[5]]
init_colorlist = colorlist
plt.rcParams["axes.prop_cycle"] = cycler(color=colorlist)

subdir_name = "tree/sim"

#chroms = range(1,25)
classification_types = ["neutral", "add", "dom", "rec", "over", "under", "full"]
sel_types = ["add", "dom", "rec", "over", "under"]
sel_types_rows = ["add", "dom", "rec", "over", "under"]
sel_strs = [.005, .01, .025, .05]
num_gens_list = [101, 251, 1001]
init_dists = ["recip", .25, .005]

# subdir_name = "tree/real/fake"
# num_gens_list = [125]
# init_dists = ["real_special"]

thinness = "real" in subdir_name

hmm_prefix_str = "" if "sim" in subdir_name else "tcec_superfinal/"
hmm_fname_str = "linapprox_nbeta_final_" if "sim" in subdir_name else ""
data_prefix_str = "" if "sim" in subdir_name else "finalfinal/"

alpha = .05
manhattan_path = Path(f"../runs/{subdir_name}/figures")

med_fit = False
for num_gens in num_gens_list:
    for init_dist in init_dists:
        #for knee in [5000, 10000, 20000, 40000, 80000]:
            if init_dist in [.005, "recip"]:
                colorlist = ["#1D6996", coolormap(1), colors[3], colors[4]]
                plt.rcParams["axes.prop_cycle"] = cycler(color=colorlist)
            else:
                plt.rcParams["axes.prop_cycle"] = cycler(color=init_colorlist)
            max_y = 0
            auc_df = np.zeros((len(sel_types), len(sel_strs)))

            ndict = {}
            ndict["sel_type"] = "neutral"
            ndict["num_gens"] = num_gens
            ndict["init_dist"] = init_dist

            neutral_filename = params_dict_to_str(**ndict)
            neutral_hmm_path = Path(f"../runs/{subdir_name}/EM/{hmm_prefix_str}{neutral_filename}_{hmm_fname_str}EM.bz2")
            with BZ2File(neutral_hmm_path) as file:
                nf = pickle.load(file)

            neutral_ll = nf["neutral_ll"]
            fig, axs = plt.subplots(1,1,figsize=(3.1,3.1), layout="constrained")
            axs.text(-.2, .97, r"$\bf{B}$", fontsize=13, transform=axs.transAxes)
            #fig, axs = plt.subplots(1, 2, figsize=(16, 8), layout="constrained")
            chi2_lr_space = np.linspace(0, 20, 500)

            axins = axs.inset_axes([.67, .11, .28, .28])
            logps = []
            labels = []
            for type_i, run_type in enumerate(sel_types):
                if init_dist in [.005, "recip"] and run_type == "rec":
                    continue
                run_EM_str = run_type if run_type not in ['over', 'under'] else 'het'
                current_bigtable_list = []
                pdict = deepcopy(ndict)
                pdict["sel_type"] = f"{run_type}"
                run_ll = nf[f"{run_EM_str}_run"]["ll_final"]
                #run_ll_2 = nf_2[f"{run_EM_str}_run"]["ll_final"]
                llr = 2*(run_ll-neutral_ll)
                #llr_2 = 2*(run_ll_2-neutral_ll_2)
                llr[llr <= 0] = 1e-12
                llr_med = np.median(llr)

                gengamma_sl_fit = gengamma(*gengamma.fit(llr[llr > llr_med] - llr_med, floc=0, fscale=1))
                chisq_sl_fit = chi2(1)

                med_p_vals = np.zeros_like(llr)
                med_p_vals[llr > llr_med] = (1 - gengamma_sl_fit.cdf(llr[llr > llr_med] - llr_med)) / 2
                med_p_vals[llr <= llr_med] = np.clip(1 - llr[llr <= llr_med] / (2 * llr_med), .5, 1)

                full_p_vals = -chisq_sl_fit.logsf(llr)/np.log(10)
                #full_p_vals_2 = -chisq_sl_fit.logsf(llr_2)/np.log(10)

                for str_i, sel_str in enumerate(sel_strs):
                    pdict["sel_str"] = sel_str
                    nn_fname = params_dict_to_str(**pdict)
                    nn_path = Path(f"../runs/{subdir_name}/EM/{hmm_prefix_str}{nn_fname}_{hmm_fname_str}EM.bz2")
                    if not nn_path.is_file():
                        print(f"{nn_path} not found")
                        sf_llr = np.zeros(nf["neutral_ll"].shape[0])
                    else:
                        with BZ2File(nn_path) as file:
                            sf = pickle.load(file)
                        sf_run_ll = sf[f"{run_EM_str}_run"]["ll_final"]
                        sf_llr = 2*(sf_run_ll-sf["neutral_ll"])
                        print(f"{nn_fname} {run_type} stats: {np.min(sf_llr):.4f} {(sf_llr < 0).sum()}")
                    #nn_llr.extend(sf_llr.tolist())
                    nn_llr = sf_llr

                    if med_fit:
                        nn_p_vals = np.zeros_like(nn_llr)
                        nn_p_vals[nn_llr > llr_med] = (1 - gengamma_sl_fit.cdf(nn_llr[nn_llr > llr_med] - llr_med)) / 2
                        nn_p_vals[nn_llr <= llr_med] = np.clip(1 - nn_llr[nn_llr <= llr_med] / (2 * llr_med), .5, 1)

                        roc_FPR, roc_TPR, auc = get_roc_and_auc(med_p_vals, nn_p_vals)
                    else:
                        nn_p_vals = -chisq_sl_fit.logsf(nn_llr)/np.log(10)
                        roc_FPR, roc_TPR, auc = get_roc_and_auc(np.power(10, -full_p_vals), np.power(10, -nn_p_vals))

                    auc_df[type_i, str_i] = f"{auc:.2f}"

                if run_type == "under":
                    continue
                logps.append(full_p_vals)
                labels.append(convert_from_abbrevs(run_EM_str, shorthet=True))
            plot_qq(axs, axins, logps, labels, legend_loc="upper left", thin=thinness)
            #axs.set_title(fr"$\chi^2(1)$ qq plot")
            #axs.set_title("Permuted data matched simulations")
            fig.savefig(Path(f"../runs/{subdir_name}/paper_figures/finalfinal/llr_plots/neutral_g{num_gens}_d{init_dist}_llr_all.pdf"), format="pdf", bbox_inches="tight")
            #fig.savefig(Path(f"../runs/{subdir_name}/paper_figures/finalfinal/llr_plots/neutral_g{num_gens}_d{init_dist}_llr_all.png"), format="png", bbox_inches="tight")
            plt.close(fig)

            big_ts_dict = {}
            big_ts_dict["num_gens"] = f"{num_gens}"
            big_ts_dict["init_dist"] = f"{init_dist}"
            big_ts_str = params_dict_to_str(**big_ts_dict)
            big_df = DataFrame(auc_df, columns=sel_strs, index=convert_from_abbrevs(sel_types_rows, shorthet=True))
            big_df = big_df.loc[~(big_df == 0).all(axis=1)]
            fig_height = 4.5 if big_df.shape[0] == len(sel_types_rows) else 3.75
            fig_width = 3.1
            fig2, axs2 = plt.subplots(1,1,figsize=(fig_width,fig_height/(8/fig_width)),layout="constrained")
            sns.heatmap(big_df, cmap="crest_r", linewidth=.8, cbar=False, fmt=".2f", annot=True, annot_kws={'fontsize':10},ax=axs2)
            axs2.text(-.57, 1.1, r"$\bf{C}$", fontsize=13, transform=axs2.transAxes)
            axs2.tick_params(axis='both', which='both', length=0, labeltop=True, labelbottom=False)
            axs2.set_xlabel(r"$\bf{s}$", fontsize=10)
            axs2.xaxis.set_label_position('top')
            axs2.set_ylabel(r"$\bf{Mode}$", fontsize=10)
            fig2.savefig(Path(f"../runs/{subdir_name}/paper_figures/finalfinal/auc_plots/{big_ts_str}_{'gengamma' if med_fit else 'chi2'}_auc_plot.pdf"), format="pdf", bbox_inches="tight")
            plt.close(fig2)