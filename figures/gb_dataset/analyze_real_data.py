import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import bz2
from scipy.stats import chi2, pearsonr
from tqdm import tqdm
from util import plot_one_snp, average_p_vals, get_1d_s_data_from_type, bh_correct, extendedFisher, windowMatrix, plot_qq, convert_from_abbrevs
from itertools import product as itprod
import pandas as pd
from matplotlib.ticker import EngFormatter
from matplotlib.colors import rgb2hex
from copy import deepcopy

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
colorlist = ["#1d6996", *[coolormap(i) for i in [1,0]], colors[3], colors[4], colors[5]]


hex_colorlist = deepcopy(colorlist)
hex_colorlist[1] = rgb2hex(colorlist[1])
hex_colorlist[2] = rgb2hex(colorlist[2])
big_colorlist = hex_colorlist*1000
full_labels = ["Additive", "Dominant", "Recessive", "Overdom.", "Underdom.", "Indet."]

subdir_name = "tree/real/really_real"
manhattan_path = Path(f"../runs/{subdir_name}/paper_figures/tcec_superfinal")
#manhattan_path = Path(f"../runs/tree/real/fake/paper_figures/linapprox/qqs")
chroms = range(1,23)
#chroms.extend(["X", "Y"])
classification_types = ["add", "dom", "rec", "het", "full"]
sig_genes = ["LCT", "LCT", "TLR10/1/6", "SLC45A2", "SLC22A4", "HLA", "TFR2"]
init_dist = np.zeros(500)


#MAGIC SAMPLE VALUES = present 92 anc 586 for SG, 504 anc for capture_only


fit_dist = chi2(1)
#full_fit_dist = chi2(2)
file_strs = ["v50.0", "v52.2", "v54.1_most_snps", "v54.1_least_snps"]

file_strs_short = [file_str[:7] for file_str in file_strs]
idx_set = []

#excluded_idxs = [9420, 15178]

# mask = np.full(73998, True)
# mask[9420] = False
# mask[15178] = False

max_ancient_ns = 0
max_present_ns = 0

#highest_idxs = np.random.default_rng(seed=5).choice(np.arange(73996), size=5, replace=False)
highest_idxs = np.array([10516, 15091, 15094, 15102, 15107, 18161, 23615, 25820, 29594, 36710, 43674, 44515, 51149, 55174, 57888, 61799])

#window = 10000

alpha = .05

primary_logp_init = 4
secondary_logp_init = 2.5

snp_window_size_init = 5
#on either side
snps_needed_init = 4
#total needed out of 10 on both sides

#p_vals_list = []
start_pos = 0
#fig, axs = plt.subplots(1, 1, figsize=(20, 12))
extra_strs = ["iter_10k", "normal_chebyshev", "numerical_beta_test", "numerical_beta_itertest", "numerical_beta_partialitertest"]
#extra_strs = []
#chrom = 5

#fig, axs = plt.subplots(1,1,figsize=(20,10),layout="constrained")
#hist_fig, hist_axs = plt.subplots(1,1,figsize=(20,20),layout="constrained")


init_strs = ["d25"]

scatter_markersize = 2
scatter_bh_width = .75


start_pos = 0
chrom_rsid_dict = {}
bias_corr_dict = {}
for genodata_type in ["capture_only"]:
    complete_agg_data_path = Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_complete_agg_data.pkl")
    with open(complete_agg_data_path, "rb") as file:
        cdata = pickle.load(file)
    p_bhs = []
    for c_type in classification_types:
        with open(Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_{c_type}_bh.pkl"), "rb") as file:
            snp_df = pickle.load(file)
        p_bhs.append(-np.log10(snp_df["p_bh"]))
    avg_bh = np.mean(p_bhs)
    all_windows = {}
    for c_type in classification_types:
        with open(Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_{c_type}_bh.pkl"), "rb") as file:
            snp_df = pickle.load(file)

        # unlog_p = np.power(10, -all_p[f"{c_type}_p"])
        windowed_p = windowMatrix(cdata["all_p"][f"{c_type}_p"], 25)
        brown_p = extendedFisher(windowed_p, standardBrown=True)
        brown_p = np.concatenate((np.zeros(26), brown_p, np.zeros(25)))
        bp_bh = -np.log10(bh_correct(np.power(10, -brown_p), alpha)[0])
        brown_p_sig_idx = np.where(brown_p > bp_bh)[0]
        p_bonferroni = .05/brown_p.shape[0]
        brown_diffs = np.diff(brown_p_sig_idx)
        brown_window_boundaries = np.concatenate(([0], np.where(brown_diffs>1)[0]))
        brown_p_windows = []
        for i in range(brown_window_boundaries.shape[0] - 1):
            brown_p_windows.append(
                brown_p_sig_idx[brown_window_boundaries[i] + 1:brown_window_boundaries[i + 1] + 1].tolist())
        brown_p_windows.append(brown_p_sig_idx[brown_window_boundaries[-1] + 1:].tolist())
        valid_windows = []
        for brown_window in brown_p_windows:
            if np.intersect1d(brown_window, snp_df["snp_idx"]).shape[0] > 0:
                if cdata["all_chrom"][brown_window[0]] == cdata["all_chrom"][brown_window[-1]]:
                    valid_windows.append(brown_window)
        for v_window in valid_windows:
            chrom = int(cdata["all_chrom"][v_window[0]])
            for zoomscatter_type in ["brown", "plain"]:
                continue
                snp_fig, snp_axs = plt.subplots(1,1,figsize=(3.1, 3.1),layout="constrained")
                if "full" in c_type and "plain" in zoomscatter_type:
                    snp_axs.plot(cdata["all_loc_per_chrom"][v_window[0]-2:v_window[-1]+2], cdata["all_p"][f"{c_type}_p"][v_window[0]-2:v_window[-1]+2], "o", markersize=1.5, color=colorlist[0], alpha=.5)
                else:
                    snp_axs.plot(cdata["all_loc_per_chrom"][v_window[0] - 2:v_window[-1] + 2],
                                 cdata["all_p"][f"{c_type}_p"][v_window[0] - 2:v_window[-1] + 2], "o", ms=scatter_markersize, color=colorlist[0], label="Raw", zorder=2.1)
                #snp_axs.text(-.2, .97, "A" if combined_idxs[snp_i] > 300000 else "B", fontsize=50, transform=snp_axs.transAxes)
                #snp_axs.axhline(-np.log10(p_bonferroni), color="b", ls="--", label=r"BF ($\alpha = .05$)")


                snp_axs.set_xlabel(f"Position (Mbp) on Chr. {chrom}")
                snp_axs.set_ylabel(r"$-\log_{10}(p)$")
                if "brown" in zoomscatter_type:
                    snp_axs.plot(cdata["all_loc_per_chrom"][v_window[0]-2:v_window[-1]+2],
                                 brown_p[v_window[0]-2:v_window[-1]+2],"^", ms=scatter_markersize, c=colorlist[1], label="Post")
                elif "full" in c_type:
                    window_mask = (snp_df["snp_idx"]>=v_window[0])&(snp_df["snp_idx"]<=v_window[-1])
                    for c_i in range(1,len(full_labels)+1):
                        color_mask = window_mask&(snp_df["classes"]==c_i)
                        if (type_sum := color_mask.sum())>0:
                            snp_axs.plot(snp_df["snp_pos"][color_mask], snp_df["p"][color_mask], "o", ms=scatter_markersize, color=colorlist[c_i-1], label=f"{full_labels[c_i-1]} ({type_sum})")
                snp_axs.axhline(-np.log10(snp_df["p_bh"]), color=colorlist[0], ls="--", lw=scatter_bh_width, label="Raw BH thresh.")
                if "brown" in zoomscatter_type:
                    snp_axs.axhline(bp_bh, color=colorlist[1], ls="--", lw=scatter_bh_width, label="Post BH thresh.")
                if zoomscatter_type=="plain" and c_type=="full":
                    snp_axs.legend(fontsize=7, labelspacing=.2, handlelength=1.5, handleheight=.5,
                               handletextpad=.4, borderpad=.2, borderaxespad=.2)
                else:
                    snp_axs.legend()
                snp_axs.ticklabel_format(axis="x",scilimits=(6,6))
                plt.setp(snp_axs.get_xaxis().get_offset_text(), visible=False)
                snp_axs.set_ylim([0, 18])
                #snp_fig.savefig(manhattan_path/f"{zoomscatter_type}_zoomscatters/{c_type}/{zoomscatter_type}_{c_type}_{v_window[0]}_correctwindows.pdf", format="pdf", bbox_inches="tight")
                if zoomscatter_type=="brown" and c_type=="add":
                    snp_axs.text(-.2, .97, r"$\bf{A}$", fontsize=13, transform=snp_axs.transAxes)
                snp_fig.savefig(
                    manhattan_path / f"{zoomscatter_type}_zoomscatters/{c_type}/{zoomscatter_type}_{c_type}_{v_window[0]}_correctwindows.pdf",
                    format="pdf", bbox_inches="tight")

                plt.close(snp_fig)
            if c_type == "dom":
                continue
                marker_types = ["o", "v", "s", "*", "P"]
                snp_fig, snp_axs = plt.subplots(1,1,figsize=(5.1,5.1),layout="constrained")
                for sc_i, subc_type in enumerate(classification_types):
                    x = cdata["all_loc_per_chrom"][v_window[0] - 2:v_window[-1] + 2]
                    len_window = x.shape[0]
                    edgecolors = np.zeros(len_window, dtype=object)
                    edgecolors[1::4] = "#0000ff"
                    edgecolors[2::4] = "#ff0000"
                    edgecolors[3::4] = "#00aa00"
                    edgecolors[::4] = "#000000"
                    snp_axs.scatter(x, cdata["all_p"][f"{subc_type}_p"][v_window[0] - 2:v_window[-1] + 2], color=colorlist[sc_i],
                                marker=marker_types[sc_i], s=scatter_markersize*7, edgecolor=edgecolors, linewidths=.3, #cmcap="plasma",
                                label=f"{convert_from_abbrevs(subc_type, shorthet=True)}")
                snp_axs.axhline(avg_bh, color="k", ls="--", lw=scatter_bh_width, label=r"BH thresh.")
                snp_axs.set_xlabel("Position (Mbp)")
                snp_axs.set_ylabel(r"$-\log_{10}(p)$")
                snp_axs.ticklabel_format(axis="x", scilimits=(6, 6))
                plt.setp(snp_axs.get_xaxis().get_offset_text(), visible=False)
                snp_axs.set_ylim([0, 18])
                snp_axs.legend(markerscale=1.25)
                snp_fig.savefig(
                    manhattan_path / f"plain_zoomscatters/combined/all_{v_window[0]}_testing.pdf",
                    format="pdf", bbox_inches="tight")
                plt.close(snp_fig)
        sw_lpos = []
        sw_rpos = []
        sw_pmax = []
        sw_spmax = []
        sw_idxmax = []
        sw_argpmax = []
        sw_rsidmax = []
        sw_chrs = []
        sw_nums = []
        sw_snps = []
        sw_raw_snps = []
        sw_raw_nums = []
        sw_type = []
        sw_ref = []
        sw_alt = []
        sw_chridxmaxs = []
        for sig_window in valid_windows:
            continue
            if "SG" in genodata_type:
                continue
            full_window = np.arange(sig_window[0], sig_window[-1]+1)
            lpos = cdata["all_loc_per_chrom"][min(sig_window)]
            rpos = cdata["all_loc_per_chrom"][max(sig_window)]
            sw_lpos.append(lpos)
            sw_rpos.append(rpos)
            window_p_vals = cdata["all_p"][f"{c_type}_p"][full_window]
            print(window_p_vals)
            sw_pmax.append(max(window_p_vals))
            window_argmax = np.argmax(window_p_vals)
            sw_idxmax.append(full_window[window_argmax])
            sw_argpmax.append(cdata["all_loc_per_chrom"][full_window[window_argmax]])
            sw_rsidmax.append(cdata["all_rsid"][full_window[window_argmax]])
            sw_ref.append(cdata["all_ref_allele"][full_window[window_argmax]])
            sw_alt.append(cdata["all_alt_allele"][full_window[window_argmax]])
            sw_spmax.append(cdata["all_s"][f"{c_type}_s"][full_window[np.argmax(window_p_vals)]])
            sw_chrom = int(cdata["all_chrom"][sig_window[0]])
            sw_chrs.append(sw_chrom)
            sw_nums.append(len(sig_window))
            raw_mask = ((snp_df["snp_chr"]==cdata["all_chrom"][sig_window[0]]) & (snp_df["snp_pos"]>=lpos) & (snp_df["snp_pos"]<=rpos))
            sw_raw_snps.append(snp_df["snp_idx"][raw_mask].tolist())
            sw_raw_nums.append(raw_mask.sum())
            sw_snps.append(sig_window)
            sw_type.append(c_type)
            sw_chridxmaxs.append(full_window[window_argmax]-cdata["all_loc"][f"chr_{sw_chrom}_idx_offset"])
        if c_type != "full":
            if c_type == "het":
                bias_corr_dict["over"] = [sp for sp in sw_spmax if sp >= 0]
                bias_corr_dict["under"] = [sp for sp in sw_spmax if sp < 0]
            else:
                bias_corr_dict[c_type] = sw_spmax
        if len(sw_lpos) > 0:
            sw_array = np.array([sw_type, sw_chrs, sw_lpos, sw_rpos, sw_raw_nums, sw_nums, sw_pmax, sw_spmax, sw_idxmax, sw_argpmax, sw_rsidmax, sw_ref, sw_alt, sw_raw_snps, sw_snps, sw_chridxmaxs]).T
            brown_windows = pd.DataFrame(sw_array, columns=["type", "chr", "lpos", "rpos", "raw_num", "num", "pmax", "spmax", "idxmax", "argpmax", "rsid_max", "ref", "alt", "raw_snps", "snps", "chridxmax"])
            with open(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_{c_type}_brown_windows.pkl", "wb") as file:
                pickle.dump(brown_windows, file)
        for plt_type in [""]:
            continue
            fig, axs = plt.subplots(1,1,figsize=(6.25,2.75), layout="constrained", dpi=1500)

            axs.plot(cdata["all_loc_all_chrom"][cdata["all_chrom"]%2==1], cdata["all_p"][f"{c_type}_p"][cdata["all_chrom"]%2==1], "o", markersize=1.5, color="#888888")
            axs.plot(cdata["all_loc_all_chrom"][cdata["all_chrom"]%2==0], cdata["all_p"][f"{c_type}_p"][cdata["all_chrom"]%2==0], "o", markersize=1.5, color="#87CFFF")
            #axs.plot(all_loc_all_chrom, brown_p, "o", c="purple", alpha=0.5)
            axs.axhline(-np.log10(snp_df["p_bh"]), ls="--", c="r", label=r"BH thresh.", lw=.75)
            axs.set_ylabel(r"$-\log_{10}(p)$")
            axs.set_xlabel("Chromosome")


            all_chr_pos_offsets = np.array([cdata["all_loc"][f"chr_{i}_pos_offset"] for i in np.arange(len(chroms)+1)+1])
            axs.xaxis.set_tick_params(length=0)
            axs.set_xticks((all_chr_pos_offsets[:-1]+all_chr_pos_offsets[1:])/2)
            chrom_labels = [str(c_i) if c_i%2 else "" for c_i in chroms]
            axs.set_xticklabels(chrom_labels)
            axs.legend()
            #axs.axhline(-np.log10(p_bonferroni), c="b")
            if "brown" in plt_type:
                axs.plot(cdata["all_loc_all_chrom"], brown_p, "o", c="purple", alpha=.1)
            axs.set_ylim([0, 18])
            axs.set_xlim([0, cdata["all_loc_all_chrom"][-1]*1.001])
            for s_i, sig_window in enumerate(valid_windows[:-1]):
                if s_i == 1:
                    continue
                axs.text(cdata["all_loc_all_chrom"][sw_idxmax[s_i]], sw_pmax[s_i], sig_genes[s_i], fontsize=9, rotation=45)
            fig.savefig(manhattan_path/f"{genodata_type}_{c_type}_{plt_type}_labelled_manhattan.png", format="png", bbox_inches="tight")
            plt.close(fig)
    continue
    fig, axs = plt.subplots(1,1,figsize=(3.1, 3.1),layout="constrained", dpi=1500)
    axins = axs.inset_axes([.67, .11, .28, .28])
    logps = [cdata['all_p'][f'{ctype}_p'] for ctype in classification_types]
    labels = convert_from_abbrevs(classification_types, shorthet=True)
    plot_qq(axs, axins, logps, labels, legend_loc="upper right", thin=True)
    fig.savefig(manhattan_path/f"all_qqs.png", format="png", bbox_inches="tight")



# if not complete_agg_data_path.is_file():
    #     for chrom in tqdm(chroms):
    #         #agg_data = {}
    #         #data_extra_str = "rev_ic_" if "special" in str(init_str) else ""
    #         #
    #         agg_path = Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_c{chrom}_agg_data.bz2")
    #         with bz2.BZ2File(agg_path, "rb") as file:
    #             agg_data = pickle.load(file)
    #         real_data_pd_path = Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_c{chrom}_data.bz2")
    #         #real_data_pd_path = Path(f"../runs/{subdir_name}/datasets/cli_test/neutral_g131_{init_str}_linapprox_{data_extra_str}pd.bz2")
    #         with bz2.BZ2File(real_data_pd_path, "rb") as file:
    #             pf = pickle.load(file)
    #
    #         real_data_path = Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_c{chrom}.csv")
    #         #real_data_path = Path(f"../runs/tree/real/fake/datasets/cli_test/neutral_g131_{init_str}_linapprox_{data_extra_str}data.csv")
    #         # #real_data_path = Path(f"../runs/")
    #         #binned_path = Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_c{chrom}_binned.csv")
    #         df = np.loadtxt(real_data_path, delimiter="\t")
    #         final_data = df[:, 2::3].astype(int)
    #         num_samples = df[:, 1::3].astype(int)
    #         sample_times = df[:, ::3].astype(int)
    #         print(np.max(sample_times))
    #         # #
    #         # binned_df = np.loadtxt(binned_path, delimiter="\t")
    #         # final_data_binned = binned_df[:, 2::3].astype(int)
    #         # num_samples_binned = binned_df[:, 1::3].astype(int)
    #         # sample_times_binned = binned_df[:, ::3].astype(int)
    #
    #
    #         # #
    #     #     em_path = Path(f"../runs/{subdir_name}/EM/tcec_superfinal/UK_v54.1_{genodata_type}_c{chrom}_EM.bz2")
    #     #     # em_path = Path(f"../runs/tree/real/fake/EM/perms/neutral_g131_dal_special_linapprox_nbeta_perm{perm_i}_ng_EM.bz2")
    #     #     with bz2.BZ2File(em_path, "rb") as file:
    #     #         hf = pickle.load(file)
    #     #
    #     #
    #     #     init_mean = hf["add_run"]["ic_dist"][0, :]/(hf["add_run"]["ic_dist"][0, :]+hf["add_run"]["ic_dist"][1, :])
    #     #     agg_data["add_init_mean"] = init_mean
    #     #     neutral_ll = hf["neutral_ll"]
    #     #     pos = pf["pos"]
    #     #     # # #
    #     #     a_ns = np.sum(num_samples, axis=1)
    #     #     a_fd = np.sum(final_data, axis=1)
    #     #     agg_data["pos"] = pos
    #     #     agg_data["snp_ids"] = pf["snp_ids"]
    #     #     agg_data["a_numsamples"] = a_ns
    #     #     agg_data["a_freq"] = a_fd / a_ns
    #     #     agg_data["a_miss"] = pf["anc_missing_frac"]
    #     #     agg_data["filter_mask"] = pf["filter_mask"]
    #     #     # #
    #     #     for classification_type in classification_types:
    #     #         # qq_fig, qq_axs = plt.subplots(1, 1, figsize=(20, 20))
    #     #         # scatter_fig, scatter_axs = plt.subplots(1, 1, figsize=(20, 20))
    #     #         #all_p_vals = np.zeros(1)
    #     #         run_ll = hf[f"{classification_type}_run"]["ll_final"][-1, :] if 'iter' in str(em_path) else hf[f"{classification_type}_run"]["ll_final"]
    #     #         llr_real = 2 * (run_ll - neutral_ll)
    #     #         llr = np.copy(llr_real)
    #     #         exit_codes = hf[f"{classification_type}_run"]["exit_codes"][agg_data["filter_mask"]]
    #     #         illegal_s = 0
    #     #         illegal_s += exit_codes[exit_codes == 2].sum()
    #     #         illegal_s += exit_codes[exit_codes == 4].sum()
    #     #         illegal_s += exit_codes[exit_codes == 12].sum()
    #     #         if illegal_s > 0:
    #     #             print(f"{classification_type}: {illegal_s} illegal")
    #     #         llr[llr <= 0] = 1e-12
    #     #     #
    #     #     #
    #     #     #
    #     #         p_vals = -fit_dist.logsf(llr)/np.log(10)
    #     #     #
    #     #         agg_data[f"{classification_type}_p_vals"] = p_vals
    #     #         agg_data[f"{classification_type}_itercount"] = hf[f"{classification_type}_run"]["itercount_hist"]
    #     #         agg_data[f"{classification_type}_s_vals"] = get_1d_s_data_from_type(hf[f"{classification_type}_run"]["s_final"], classification_type)
    #     #         if not np.all(np.isfinite(p_vals)):
    #     #             print(f"c {chrom} {classification_type} illegal stuff happening")
    #     #
    #     #     with bz2.BZ2File(agg_path, "wb") as file:
    #     #         pickle.dump(agg_data, file)
    #     #     # continue
    #     #     # plt.close(fig)
    #     #     #
    #     #     # all_p_vals = np.concatenate((all_p_vals, p_vals))
    #     #     # #
    #     #
    #         for c_type in classification_types:
    #             #all_s[f"{classification_type}_s"] = np.concatenate((all_s[f"{classification_type}_s"], agg_data[f"{classification_type}_s_vals"][agg_data["filter_mask"]]))
    #             #all_p["add_p_perm"] = np.concatenate((all_p["add_p_perm"], agg_data["add_p_vals"][agg_data["filter_mask"]]))
    #             all_p[f"{c_type}_p"] = np.concatenate((all_p[f"{c_type}_p"], agg_data[f"{c_type}_p_vals"][agg_data["filter_mask"]]))
    #             all_s[f"{c_type}_s"] = np.concatenate(
    #                 (all_s[f"{c_type}_s"], agg_data[f"{c_type}_s_vals"][agg_data["filter_mask"]]))
    #             if c_type == classification_types[0]:
    #                 all_loc_all_chrom = np.concatenate((all_loc_all_chrom, agg_data["pos"][agg_data["filter_mask"]].astype(np.int64)+start_pos))
    #                 all_loc_per_chrom = np.concatenate((all_loc_per_chrom, agg_data["pos"][agg_data["filter_mask"]]))
    #                 start_pos += np.max(agg_data["pos"])
    #                 all_loc[f"chr_{chrom}"] = agg_data["pos"][agg_data["filter_mask"]]
    #                 # all_maf[f"chr_{chrom}"] = agg_data["a_freq"][agg_data["filter_mask"]]
    #                 # all_maf[f"chr_{chrom}"] = np.minimum(all_maf[f"chr_{chrom}"], 1-all_maf[f"chr_{chrom}"])
    #                 all_loc[f"chr_{chrom+1}_offset"] = all_loc[f"chr_{chrom}_offset"] + all_loc[f"chr_{chrom}"].shape[0]
    #                 #all_fd = np.concatenate((all_fd, final_data_binned[agg_data["filter_mask"], :]))
    #                 #all_ns = np.concatenate((all_ns, num_samples_binned[agg_data["filter_mask"], :]))
    #                 #all_st = np.concatenate((all_st, sample_times_binned[agg_data["filter_mask"], :]))
    #                 all_missingness = np.concatenate((all_missingness, agg_data["a_miss"][agg_data["filter_mask"]]))
    #                 all_means = np.concatenate((all_means, agg_data["add_init_mean"][agg_data["filter_mask"]]))
    #                 all_rsid = np.concatenate((all_rsid, agg_data["snp_ids"][agg_data["filter_mask"]]))
    #                 all_chrom = np.concatenate((all_chrom, np.zeros(agg_data["filter_mask"].sum())+chrom))
    #                 if chrom == 1:
    #                     all_rsid = all_rsid[1:]
    #                     all_chrom = all_chrom[1:]
    #                     all_loc_all_chrom = all_loc_all_chrom[1:]
    #                     all_loc_per_chrom = all_loc_per_chrom[1:]
    #                     all_missingness = all_missingness[1:]
    #                     all_means = all_means[1:]
    #             if chrom == 1:
    #                 all_p[f"{c_type}_p"] = all_p[f"{c_type}_p"][1:]
    #                 all_s[f"{c_type}_s"] = all_s[f"{c_type}_s"][1:]
    #                 #all_fd = all_fd[1:, :]
    #                 #all_ns = all_ns[1:, :]
    #                 #all_st = all_st[1:, :]
    #     #
    #     # for classification_type in classification_types:
    #     #     data_to_save = {}
    #     #     p_bh, bh_locs = bh_correct(np.power(10, -all_p[f"{classification_type}_p"]), alpha, yekutieli=False)
    #     #     data_to_save["snp_idx"] = bh_locs
    #     #     data_to_save["snp_pos"] = all_loc_per_chrom[bh_locs]
    #     #     data_to_save["snp_rsid"] = all_rsid[bh_locs]
    #     #     data_to_save["snp_chr"] = all_chrom[bh_locs]
    #     #     data_to_save["p"] = all_p[f"{classification_type}_p"][bh_locs]
    #     #     data_to_save["p_bh"] = p_bh
    #     #
    #     #     with open(Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_{classification_type}_bh.pkl"), "wb") as file:
    #     #         pickle.dump(data_to_save, file)
    #     #
    #     #
    #     cdata = {}
    #     cdata["all_p"] = all_p
    #     cdata["all_s"] = all_s
    #     cdata["all_loc_all_chrom"] = all_loc_all_chrom
    #     cdata["all_loc_per_chrom"] = all_loc_per_chrom
    #     cdata["all_chrom"] = all_chrom
    #     cdata["all_rsid"] = all_rsid
    #     cdata["all_loc"] = all_loc
    #     cdata["all_missingness"] = all_missingness
    #     cdata["all_means"] = all_means
    #
    #     with open(complete_agg_data_path, "wb") as file:
    #         pickle.dump(cdata, file)
