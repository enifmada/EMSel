import bz2
import numpy as np
from core import HMM
from scipy.stats import norm
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse
import allel
from pandas import read_csv
from emsel_util import vcf_to_useful_format
from joblib import Parallel, delayed

def run_one_s(iter_hmm, obs_counts, nts, sample_locs, loc, tol, max_iter, min_init_val=1e-8, min_ic = 5):
    iter_hmm.update_internals_from_datum(obs_counts, nts, sample_locs)
    iter_hmm.s1 = iter_hmm.s1_init
    iter_hmm.s2 = iter_hmm.s2_init
    iter_hmm.a = iter_hmm.a_init
    iter_hmm.init_state = iter_hmm.init_init_state
    return iter_hmm.compute_one_s(loc, tol, max_iter, min_init_val=min_init_val, min_ic=min_ic)


parser = argparse.ArgumentParser()
parser.add_argument("input_path", type=argparse.FileType("rb"), help="path to input dataset")
parser.add_argument("output_path", type=argparse.FileType("wb"), help="path to output file")
parser.add_argument("-ytg", "--years_to_gen", type=float, default=1, help="years per generation in VCF or CSV")
parser.add_argument("-nc", "--num_cores", type=int, default=1, help="number of CPU cores to parallelize over")
parser.add_argument("-ns", "--num_states", type=int, help="number of approx states in HMM", default=500)
parser.add_argument("--s_init", type=float, nargs=2, default=[0., 0.], help="vector of initial s value")
parser.add_argument("-ic", "--init_cond", default="uniform", help="initial condition to use")
parser.add_argument("-t", "--tol", type=float, default=1e-3, help="ll_i - ll_(i-1) < tol stops the run")
parser.add_argument("-m", "--maxiter", type=int, default=2000, help="maximum number of iterations")
parser.add_argument("-Ne", type=int, default=20000, help="effective population size for the HMM")
parser.add_argument("-mu", type=float, default=1.25e-8, help="mutation rate")
parser.add_argument("--hidden_interp", default="linear", help="interpolation of the hidden states (linear vs. Chebyshev nodes for now)")
parser.add_argument("--ic_update_type", default="beta", help="type of init cond estimation")
parser.add_argument("--progressbar", action="store_true", help="adds a tqdm progress bar")
parser.add_argument("--ic_dict", nargs='*', help="initial condition dictionary")
parser.add_argument("--update_types", default="all", nargs='*', help="strings of update types to run")
parser.add_argument("--min_itercount", type=int, default=5, help="minimum number of EM iterations before terminating")
parser.add_argument("--min_init_val", type=float, default=1e-8, help="minimum value of an init state probability")
parser.add_argument("--save_csv", action="store_true", help="if inputting a VCF, save a CSV to future reduce pre-processing time")
parser.add_argument("--sample_file", type=argparse.FileType("rb"), help="sample times file (if input = VCF)")
parser.add_argument("--sample_cols", type=str, nargs=2, default=["Genetic_ID","Date_mean"], help="names of the ID and dates columns in the sample times file (if input = VCF)")

args = parser.parse_args()


hmm_dd = {}
hmm_dd["approx_states"] = args.num_states
hmm_dd["s_init"] = args.s_init
hmm_dd["init_cond"] = args.init_cond
hmm_dd["tol"] = args.tol
hmm_dd["ytg"] = args.years_to_gen
hmm_dd["max_iter"] = args.maxiter
hmm_dd["Ne"] = args.Ne
hmm_dd["mu"] = args.mu
hmm_dd["hidden_interp"] = args.hidden_interp
hmm_dd["ic_update_type"] = args.ic_update_type
hmm_dd["update_types"] = args.update_types
hmm_dd["ic_dict"] = {}
hmm_dd["min_ic"] = args.min_itercount
hmm_dd["min_init_val"] = args.min_init_val

if args.ic_dict is not None:
    for ic_pair in args.ic_dict:
        k, v = ic_pair.split('=')
        try:
            hmm_dd["ic_dict"][k] = float(v)
        except:
            hmm_dd["ic_dict"][k] = v

hmm_path = Path(args.output_path.name)
pd_path = Path(args.input_path.name)
print(pd_path)
print(pd_path.suffix)
num_cores = args.num_cores

if not pd_path.is_file():
    print(f"no params dict: {pd_path}")
    raise ValueError

hmm_data = {}
if pd_path.suffix == ".csv":
    pf = np.loadtxt(pd_path, delimiter="\t")
    hmm_data["final_data"] = pf[:, 2::3].astype(int)
    hmm_data["num_samples"] = pf[:, 1::3].astype(int)
    hmm_data["sample_times"] = (pf[:, ::3] / hmm_dd["ytg"]).astype(int)
elif pd_path.suffix == ".vcf":
    if pd_path.with_suffix(".csv").is_file():
        print("CSV already generated!")
        pf = np.loadtxt(pd_path.with_suffix(".csv"), delimiter="\t")
        hmm_data["final_data"] = pf[:, 2::3].astype(int)
        hmm_data["num_samples"] = pf[:, 1::3].astype(int)

        #don't divide by YTG here because the VCF -> CSV process already does this
        hmm_data["sample_times"] = pf[:, ::3].astype(int)
    else:
        print("processing VCF...")
        if not (args.sample_file and args.sample_cols):
            raise ValueError("No sample file (or columns therein) specified!")
        vcf_file = allel.read_vcf(str(pd_path))
        vcf_dates = read_csv(args.sample_file.name, usecols=args.sample_cols, sep="\t").to_numpy()
        full_array = vcf_to_useful_format(vcf_file, vcf_dates, years_per_gen = hmm_dd["ytg"])
        hmm_data["final_data"] = full_array[:, 2::3]
        hmm_data["num_samples"] = full_array[:, 1::3]
        hmm_data["sample_times"] = full_array[:, ::3]
        if args.save_csv:
            np.savetxt(pd_path.with_suffix(".csv"), full_array, delimiter="\t", fmt="%d")
        print("done processing VCF")

hmm_data["final_data"] = hmm_data["final_data"][:50, :]
hmm_data["num_samples"] = hmm_data["num_samples"][:50, :]
hmm_data["sample_times"] = hmm_data["sample_times"][:50, :]

all_update_types = ["neutral", "add", "dom", "rec", "het", "full"]
if hmm_dd["update_types"] == "all" or hmm_dd["update_types"] == ["all"]:
    update_types = all_update_types
else:
    if len([update_i for update_i in hmm_dd["update_types"] if update_i not in [*all_update_types, "fixed"]]) > 0:
        raise ValueError("Invalid update type specified!")
    update_types = hmm_dd["update_types"]
for update_i, update_type in enumerate(update_types):
    print(f"{update_type}!")
    if num_cores > 1:
        parallel_loop = tqdm(range(hmm_data["final_data"].shape[0])) if args.progressbar else range(hmm_data["final_data"].shape[0])
        with Parallel(n_jobs=num_cores) as parallel:
            iter_hmm = HMM(hmm_dd["approx_states"], hmm_dd["Ne"], hmm_dd["mu"],hmm_dd["s_init"],
                        init_cond = hmm_dd["init_cond"], hidden_interp = hmm_dd["hidden_interp"], **hmm_dd["ic_dict"])
            iter_hmm.update_type = update_type
            iter_hmm.update_func, iter_hmm.update_func_args = iter_hmm.get_update_func(update_type, {})
            iter_hmm.init_update_type = hmm_dd["ic_update_type"]
            iter_hmm.init_update_func, iter_hmm.init_params_to_state_func, iter_hmm.init_update_size = iter_hmm.get_init_update_func(hmm_dd["ic_update_type"])
            res = parallel(delayed(run_one_s)(iter_hmm, hmm_data["final_data"][i], hmm_data["num_samples"][i], hmm_data["sample_times"][i], i, hmm_dd["tol"], hmm_dd["max_iter"], hmm_dd["min_init_val"], hmm_dd["min_ic"]) for i in parallel_loop)
        hmm_dict = {
            "s_final": np.array([rp[1] for rp in res]).T,
            "ll_hist": np.array([rp[2] for rp in res]).T,
            "ic_dist": np.array([rp[3] for rp in res]).T,
            "itercount_hist": np.array([rp[4] for rp in res]),
            "exit_codes": np.array([rp[5] for rp in res]),
        }
    else:
        iter_hmm = HMM(hmm_dd["approx_states"], hmm_dd["Ne"], hmm_dd["mu"], np.array(hmm_dd["s_init"]),init_cond=hmm_dd["init_cond"], hidden_interp=hmm_dd["hidden_interp"], **hmm_dd["ic_dict"])
        iter_hmm.update_type = update_type
        iter_hmm.update_func, iter_hmm.update_func_args = iter_hmm.get_update_func(update_type, {})
        s_hist, s_final, ll_hist, ic_dist, itercount_hist, exit_codes = iter_hmm.compute_s(hmm_data["final_data"], hmm_data["num_samples"], hmm_data["sample_times"],
                                   update_type, hmm_dd["ic_update_type"], hmm_dd["tol"], hmm_dd["max_iter"],progressbar=args.progressbar)
        hmm_dict = {
            "s_final": s_final,
            "ll_hist": ll_hist,
            "ic_dist": ic_dist,
            "itercount_hist": itercount_hist,
            "exit_codes": exit_codes
        }
    hmm_dict["ll_final"] = np.array([hmm_dict["ll_hist"][hmm_dict["itercount_hist"][i], i] for i in range(hmm_dict["itercount_hist"].shape[0])])
    hmm_dict["ll_final_valid"] = np.where(hmm_dict["ll_final"] != hmm_dict["ll_hist"][0, :],
                                              hmm_dict["ll_final"], -np.inf)

    if update_type == "neutral":
        hmm_dd["neutral_ll"] = np.array([hmm_dict["ll_hist"][hmm_dict["itercount_hist"][i], i] for i in range(hmm_dict["itercount_hist"].shape[0])])
        hmm_dd["neutral_ic"] = hmm_dict["ic_dist"]
        hmm_dd["neutral_itercount"] = hmm_dict["itercount_hist"]
    else:
        hmm_dd[f"{update_type}_run"] = hmm_dict
    del hmm_dict["ll_hist"]
    print(hmm_dict["ll_final"])
with bz2.BZ2File(hmm_path, "wb") as file:
    pickle.dump(hmm_dd, file)