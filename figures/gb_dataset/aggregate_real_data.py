import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import bz2
from scipy.stats import chi2, pearsonr
from tqdm import tqdm
from util import plot_one_snp, average_p_vals, get_1d_s_data_from_type, bh_correct, extendedFisher, windowMatrix, get_llg_array, get_llgka_array, classify_full_run, full_bh_procedure
from itertools import product as itprod
import pandas as pd

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

subdir_name = "tree/real/really_real"
manhattan_path = Path(f"../runs/{subdir_name}/paper_figures/tcec_superfinal")
#manhattan_path = Path(f"../runs/tree/real/fake/paper_figures/linapprox/qqs")
chroms = range(1,23)
#chroms.extend(["X", "Y"])
classification_types = ["add", "dom", "het", "rec"]

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

onep_types = ["add", "dom", "rec"]
init_strs = ["d25"]

all_s = {}
all_p = {}
for c_type in classification_types:
    all_p[f"{c_type}_p"] = np.array([0])
    all_s[f"{c_type}_s"] = np.array([0])
all_loc = {}
all_maf = {}
all_loc["chr_1_idx_offset"] = 0
all_loc["chr_1_pos_offset"] = 0
all_rsid = np.array([0])
all_chrom = np.array([0])
all_loc_all_chrom = np.array([0], dtype=np.int64)
all_loc_per_chrom = np.array([0])
all_fd = np.zeros((1, 5))
all_ns = np.zeros((1, 5))
all_st = np.zeros((1, 5))
all_llg_array = np.zeros((1,len(onep_types) + 4))
all_ref_allele = np.array([0])
all_alt_allele = np.array([0])

start_pos = 0
all_missingness = np.zeros(1)
all_means = np.zeros(1)

gengamma_save_path = Path("../runs/tree/real/really_real/datasets/tcec_superfinal/gengamma_params.pkl")

total_snps = 0
#chrom = 1

with open(gengamma_save_path, "rb") as file:
    gengamma_dict = pickle.load(file)
for genodata_type in ["capture_only"]:
    complete_agg_data_path = Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_complete_agg_data.pkl")
    for chrom in tqdm(chroms):
        agg_data = {}
        real_data_pd_path = Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_c{chrom}_data.bz2")

        #real_data_pd_path = Path(f"../runs/{subdir_name}/datasets/cli_test/neutral_g131_{init_str}_linapprox_{data_extra_str}pd.bz2")
        with bz2.BZ2File(real_data_pd_path, "rb") as file:
            pf = pickle.load(file)

        real_data_path = Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_c{chrom}.csv")
        #real_data_path = Path(f"../runs/tree/real/fake/datasets/cli_test/neutral_g131_{init_str}_linapprox_{data_extra_str}data.csv")
        # #real_data_path = Path(f"../runs/")
        #binned_path = Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_c{chrom}_binned.csv")
        df = np.loadtxt(real_data_path, delimiter="\t")
        final_data = df[:, 2::3].astype(int)
        num_samples = df[:, 1::3].astype(int)
        sample_times = df[:, ::3].astype(int)
        print(np.max(sample_times))

        assert pf["filter_mask"].shape[0] == final_data.shape[0]

        em_path = Path(f"../runs/{subdir_name}/EM/tcec_superfinal/UK_v54.1_{genodata_type}_c{chrom}_EM.bz2")
        with bz2.BZ2File(em_path, "rb") as file:
            hf = pickle.load(file)
        init_mean = hf["add_run"]["ic_dist"][0, :]/(hf["add_run"]["ic_dist"][0, :]+hf["add_run"]["ic_dist"][1, :])
        agg_data["add_init_mean"] = init_mean
        neutral_ll = hf["neutral_ll"]
        pos = pf["pos"]
        # # #
        a_ns = np.sum(num_samples, axis=1)
        a_fd = np.sum(final_data, axis=1)
        agg_data["pos"] = pos
        agg_data["snp_ids"] = pf["snp_ids"]
        agg_data["a_numsamples"] = a_ns
        agg_data["a_freq"] = a_fd / a_ns
        agg_data["a_miss"] = pf["anc_missing_frac"]
        agg_data["filter_mask"] = pf["filter_mask"]
        agg_data["ref_allele"] = pf["ref_allele"]
        agg_data["alt_allele"] = pf["alt_allele"]
        if chrom == 2:
            print(f"full s vals at LCT lead SNP: {hf['full_run']['s_final'][:, pf['filter_mask']][:,36503]}")
            raise Error
        temp_llg_array = get_llg_array(hf, onep_types, classify_full_run(hf["full_run"]["s_final"])[0])
        #temp_llg_array = temp_llg_array
        all_llg_array = np.vstack((all_llg_array, temp_llg_array[agg_data["filter_mask"], :]))
        for classification_type in classification_types:
            run_ll = hf[f"{classification_type}_run"]["ll_final"][-1, :] if 'iter' in str(em_path) else hf[f"{classification_type}_run"]["ll_final"]
            llr_real = 2 * (run_ll - neutral_ll)
            llr = np.copy(llr_real)
            exit_codes = hf[f"{classification_type}_run"]["exit_codes"][agg_data["filter_mask"]]
            illegal_s = 0
            illegal_s += exit_codes[exit_codes == 2].sum()
            illegal_s += exit_codes[exit_codes == 4].sum()
            illegal_s += exit_codes[exit_codes == 12].sum()
            if illegal_s > 0:
                print(f"{classification_type}: {illegal_s} illegal")
            llr[llr <= 0] = 1e-12
            p_vals = -fit_dist.logsf(llr)/np.log(10)
            agg_data[f"{classification_type}_p_vals"] = p_vals
            agg_data[f"{classification_type}_itercount"] = hf[f"{classification_type}_run"]["itercount_hist"]
            agg_data[f"{classification_type}_s_vals"] = hf[f"{classification_type}_run"]["s_final"]
            if not np.all(np.isfinite(p_vals)):
                print(f"c {chrom} {classification_type} illegal stuff happening")
        for c_type in classification_types:
            #all_s[f"{classification_type}_s"] = np.concatenate((all_s[f"{classification_type}_s"], agg_data[f"{classification_type}_s_vals"][agg_data["filter_mask"]]))
            #all_p["add_p_perm"] = np.concatenate((all_p["add_p_perm"], agg_data["add_p_vals"][agg_data["filter_mask"]]))
            all_p[f"{c_type}_p"] = np.concatenate((all_p[f"{c_type}_p"], agg_data[f"{c_type}_p_vals"][agg_data["filter_mask"]]))
            all_s[f"{c_type}_s"] = np.concatenate(
                (all_s[f"{c_type}_s"], agg_data[f"{c_type}_s_vals"][agg_data["filter_mask"]]))
            if c_type == classification_types[0]:
                all_loc_all_chrom = np.concatenate((all_loc_all_chrom, agg_data["pos"][agg_data["filter_mask"]].astype(np.int64)+start_pos))
                all_loc_per_chrom = np.concatenate((all_loc_per_chrom, agg_data["pos"][agg_data["filter_mask"]]))
                start_pos += np.max(agg_data["pos"])
                all_loc[f"chr_{chrom}"] = agg_data["pos"][agg_data["filter_mask"]]
                # all_maf[f"chr_{chrom}"] = agg_data["a_freq"][agg_data["filter_mask"]]
                # all_maf[f"chr_{chrom}"] = np.minimum(all_maf[f"chr_{chrom}"], 1-all_maf[f"chr_{chrom}"])
                all_loc[f"chr_{chrom+1}_idx_offset"] = all_loc[f"chr_{chrom}_idx_offset"] + all_loc[f"chr_{chrom}"].shape[0]
                all_loc[f"chr_{chrom+1}_pos_offset"] = all_loc[f"chr_{chrom}_pos_offset"] + all_loc[f"chr_{chrom}"][-1]
                #all_fd = np.concatenate((all_fd, final_data_binned[agg_data["filter_mask"], :]))
                #all_ns = np.concatenate((all_ns, num_samples_binned[agg_data["filter_mask"], :]))
                #all_st = np.concatenate((all_st, sample_times_binned[agg_data["filter_mask"], :]))
                all_missingness = np.concatenate((all_missingness, agg_data["a_miss"][agg_data["filter_mask"]]))
                all_means = np.concatenate((all_means, agg_data["add_init_mean"][agg_data["filter_mask"]]))
                all_rsid = np.concatenate((all_rsid, agg_data["snp_ids"][agg_data["filter_mask"]]))
                all_chrom = np.concatenate((all_chrom, np.zeros(agg_data["filter_mask"].sum())+chrom))
                all_ref_allele = np.concatenate((all_ref_allele, agg_data["ref_allele"][agg_data["filter_mask"]]))
                all_alt_allele = np.concatenate((all_alt_allele, agg_data["alt_allele"][agg_data["filter_mask"]]))

                if chrom == 1:
                    all_rsid = all_rsid[1:]
                    all_chrom = all_chrom[1:]
                    all_loc_all_chrom = all_loc_all_chrom[1:]
                    all_loc_per_chrom = all_loc_per_chrom[1:]
                    all_missingness = all_missingness[1:]
                    all_means = all_means[1:]
                    all_ref_allele = all_ref_allele[1:]
                    all_alt_allele = all_alt_allele[1:]
                    MAF_THRESH = pf["maf_thresh"]
                    MISSING_THRESH = pf["missing_thresh"]
            if chrom == 1:
                all_p[f"{c_type}_p"] = all_p[f"{c_type}_p"][1:]
                all_s[f"{c_type}_s"] = all_s[f"{c_type}_s"][1:]
                #all_fd = all_fd[1:, :]
                #all_ns = all_ns[1:, :]
                #all_st = all_st[1:, :]


    print("done")
    all_llg_array = all_llg_array[1:, :]
    all_llgka_array = get_llgka_array(all_llg_array[:, :, np.newaxis], k=gengamma_dict["k_opt"], alpha=0)
    p_full_bh, full_p_vals, full_classes = full_bh_procedure([all_llgka_array], gengamma_dict["gengamma_fit"], gengamma_dict["lr_shift"], alpha, bh=True)
    full_p_vals = -np.log10(full_p_vals[0].flatten())
    full_classes = full_classes[0].flatten()
    full_data_to_save = {}
    bh_locs = np.where(full_classes>0)[0]
    full_data_to_save["snp_idx"] = bh_locs
    full_data_to_save["snp_pos"] = all_loc_per_chrom[bh_locs]
    full_data_to_save["snp_rsid"] = all_rsid[bh_locs]
    full_data_to_save["snp_chr"] = all_chrom[bh_locs]
    full_data_to_save["p"] = full_p_vals[bh_locs]
    full_data_to_save["p_bh"] = p_full_bh
    full_data_to_save["classes"] = full_classes[bh_locs]

    with open(Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_full_bh.pkl"),"wb") as file:
        pickle.dump(full_data_to_save, file)


    all_p["full_p"] = full_p_vals
    all_s["full_s"] = all_s["add_s"]

    for classification_type in classification_types:
        data_to_save = {}
        p_bh, bh_locs = bh_correct(np.power(10, -all_p[f"{classification_type}_p"]), alpha, yekutieli=False)
        data_to_save["snp_idx"] = bh_locs
        data_to_save["snp_pos"] = all_loc_per_chrom[bh_locs]
        data_to_save["snp_rsid"] = all_rsid[bh_locs]
        data_to_save["snp_chr"] = all_chrom[bh_locs]
        data_to_save["p"] = all_p[f"{classification_type}_p"][bh_locs]
        data_to_save["p_bh"] = p_bh

        with open(Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_{classification_type}_bh.pkl"), "wb") as file:
            pickle.dump(data_to_save, file)

    cdata = {}
    cdata["all_p"] = all_p
    cdata["all_s"] = all_s
    cdata["all_loc_all_chrom"] = all_loc_all_chrom
    cdata["all_loc_per_chrom"] = all_loc_per_chrom
    cdata["all_chrom"] = all_chrom
    cdata["all_rsid"] = all_rsid
    cdata["all_loc"] = all_loc
    cdata["all_missingness"] = all_missingness
    cdata["all_means"] = all_means
    cdata["all_ref_allele"] = all_ref_allele
    cdata["all_alt_allele"] = all_alt_allele
    cdata["maf_thresh"] = MAF_THRESH
    cdata["missing_thresh"] = MISSING_THRESH

    with open(complete_agg_data_path, "wb") as file:
        pickle.dump(cdata, file)

    sim_data = {}
    sim_data["all_missingness"] = cdata["all_missingness"]
    sim_data["all_means"] = cdata["all_means"]
    sim_data["maf_thresh"] = MAF_THRESH
    sim_data["missing_thresh"] = MISSING_THRESH
    with open(Path(f"../runs/{subdir_name}/datasets/tcec_superfinal/UK_v54.1_{genodata_type}_sim_agg_data.pkl"), "wb") as file:
        pickle.dump(sim_data, file)