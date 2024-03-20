import numpy as np
from emsel_util import params_dict_to_str, generate_data, get_sel_coeffs_from_type
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product as itprod
import bz2
import argparse


def float_or_str(val):
    try:
        return float(val)
    except:
        return val
parser = argparse.ArgumentParser()
parser.add_argument("output_directory", type=str, help="path to output directory")
parser.add_argument("-n", "--num_sims", type=int, default=100, help="number of replicates")
parser.add_argument("-s", "--sel_coeffs", nargs="+", type=float, default=[.005, .01, .025, .05], help="selection coefficients to simulate")
parser.add_argument("--sel_types", nargs="+", type=str, default=["neutral", "add", "dom", "rec", "over", "under"], help="types of selection to simulate. valid types are 'neutral', 'add', 'dom', 'rec', 'over', 'under', for now.")
parser.add_argument("-g", "--num_gens", nargs="+", type=int, default=[101, 251, 1001], help="number of generations to simulate")
parser.add_argument("-ic", "--init_cond", nargs="+", type=float_or_str, default=[.005, .25, "recip"], help="initial conditions to simulate")
parser.add_argument("-ns", "--num_samples", type=int, default=50, help="number of samples to draw at each sampling timepoint")
parser.add_argument("-st", "--sampling_times", type=int, default=11, help="number of times to draw samples")
parser.add_argument("-Ne", type=int, default=10000, help="effective population size")
parser.add_argument("--data_matched", type=str, nargs=2, default=["", ""], help="input the path to an initial frequency + missingness (sim_data) file and a sampling table, will override -g, -ic, -ns and -st to initialize and sample according to the table")
parser.add_argument("--seed", type=int, default=5, help="seed")
parser.add_argument("--save_plots", action="store_true", help="save plots of all of the replicate trajectories")
parser.add_argument("--suffix", type=str, default="", help="file names suffix to differentiate")
args = parser.parse_args()

args_dict = {}
args_dict["s_list"] = args.sel_coeffs
args_dict["g_list"] = args.num_gens
args_dict["sel_types_list"] = []
for arg_sel_type in args.sel_types:
    if arg_sel_type not in ["neutral", "add", "dom", "rec", "over", "under"]:
        raise TypeError(f"invalid selection type: {arg_sel_type}")
    args_dict["sel_types_list"].append(arg_sel_type)
args_dict["ic_list"] = args.init_cond
args_dict["seed"] = args.seed

if args.suffix != "":
    args.suffix = args.suffix + "_"

pd = {
    "sample_times": args.sampling_times,
    "num_samples": args.num_samples,
    "Ne": args.Ne,
    "num_sims": args.num_sims,
}

if args.data_matched[0] != "":
    sampling_matrix = np.loadtxt(Path(f"{args.data_matched[1]}"), skiprows=1, dtype=int)
    with open(Path(f"{args.data_matched[0]}"), "rb") as file:
        real_data_file = pickle.load(file)
    sampling_matrix[:, 0] = sampling_matrix[-1, 0] - sampling_matrix[:, 0]
    sampling_matrix = np.flip(sampling_matrix, axis=0)
    args_dict["g_list"] = [sampling_matrix[-1, 0] + 1]
    args_dict["ic_list"] = ["real_special"]
    args_dict["ic_path"] = args.data_matched[0]
    args_dict["table_path"] = args.data_matched[1]

temp_seed = args_dict["seed"]
args_save_path = Path(f"{args.output_directory}/args{args.suffix}.pkl")
for sel_str, sel_type, init_dist, num_gens in itprod(args_dict["s_list"], args_dict["sel_types_list"], args_dict["ic_list"], args_dict["g_list"]):
    temp_seed += 1
    if sel_type == "under" and init_dist == .005:
        continue
    pdict = {}
    pdict["sel_type"] = sel_type
    pdict["num_gens"] = num_gens
    if sel_type != "neutral":
        pdict["sel_str"] = sel_str
    pdict["init_dist"] = init_dist

    exp_name = params_dict_to_str(**pdict)
    params_filename = Path(f"{args.output_directory}/{exp_name}_{args.suffix}pd.bz2")
    data_filename = Path(f"{args.output_directory}/{exp_name}_{args.suffix}data.csv")

    s1, s2 = get_sel_coeffs_from_type(sel_type, sel_str)
    pd["s1_true"] = s1
    pd["s2_true"] = s2
    pd["num_gens"] = num_gens
    pd["p_init"] = init_dist
    pd["seed"] = temp_seed
    pd["exp_name"] = exp_name
    pd["survive_only"] = True
    pd["sel_type"] = sel_type
    pd["small_s"] = True
    if args.data_matched[0] == "":
        pd["init_cond"] = "recip" if init_dist == "recip" else "delta"
    else:
        pd["init_cond"] = "real_special"

    if args.data_matched[0] != "":
        pd["real_data_file"] = real_data_file
        pd["sampling_matrix"] = sampling_matrix

    if sel_type == "neutral" and params_filename.is_file():
        continue
    data_dict = generate_data(pd)

    data_dict["obs_counts"] = data_dict["obs_counts"].astype(int)
    data_dict["nt"] = data_dict["nt"].astype(int)

    print(f"Generated replicates: {exp_name}")

    if data_dict["true_data"].shape[0] < pd["num_sims"]:
        continue

    pd["true_data"] = data_dict["true_data"]
    pd["p_inits"] = data_dict["p_inits"]
    if args.save_plots:
        fig, axs = plt.subplots(1,1,figsize=(20,20))
        axs.plot(data_dict["true_data"].T)
        axs.plot(np.mean(data_dict["true_data"], axis=0), color="k", lw=2)
        fig.savefig(Path(f"{args.output_directory}/{exp_name}_{args.suffix}_plot.png"), bbox_inches="tight")
        plt.close(fig)

    data_csv = np.zeros((data_dict["true_data"].shape[0], data_dict["sample_locs"].shape[0]*3))

    data_csv[:, ::3] = data_dict["sample_locs"]
    data_csv[:, 1::3] = data_dict["nt"]
    data_csv[:, 2::3] = data_dict["obs_counts"]
    if args.data_matched[0] != "":
        del pd["real_data_file"]
    with bz2.BZ2File(params_filename, "wb") as file:
        pickle.dump(pd, file)
    np.savetxt(data_filename, data_csv, delimiter="\t", fmt="%d")
with open(args_save_path, "wb") as file:
    pickle.dump(args_dict, file)




