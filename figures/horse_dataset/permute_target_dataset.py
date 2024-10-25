import numpy as np
from pathlib import Path
from tqdm import tqdm
from emsel.emsel_util import params_dict_to_str


####### MODIFY

num_gens_list = [125]
init_dists = ["real_special"]
#num_gens_list = [101, 251, 1001]
#init_dists = [.005, .25, "recip"]

data_dir = "data"
num_perms = 1000

midfix_str = ""

###### DO NOT MODIFY

rng = np.random.default_rng(5)
for suffix in ["asip", "mc1r"]:
    df_path = Path(f"{data_dir}/horse_data_{suffix}.csv")
    df = np.loadtxt(df_path, delimiter="\t")
    new_df_idxs = np.zeros(num_perms, dtype=int)
    new_df = df[new_df_idxs, :].copy()
    final_data = new_df[:, 2::3].astype(int)
    num_samples = new_df[:, 1::3].astype(int)
    sample_times = new_df[:, ::3].astype(int)
    new_final_data = np.zeros_like(final_data)
    alt_df_path = Path(f"{data_dir}/horse_data_{suffix}_permuted.csv")
    for s_i in np.arange(sample_times.shape[0]):
        samples = np.zeros(np.sum(num_samples[s_i, :]))
        new_sample_times = np.zeros_like(samples)
        samples_counter = 0
        for st_i in np.arange(num_samples.shape[1]):
            samples_time_i = num_samples[s_i, st_i]
            hits_time_i = final_data[s_i, st_i]
            samples[samples_counter:samples_counter+hits_time_i] += 1
            new_sample_times[samples_counter:samples_counter+samples_time_i] = sample_times[s_i, st_i]
            samples_counter += samples_time_i
        rng.shuffle(new_sample_times)
        for st_i in np.arange(num_samples.shape[1]):
            new_final_data[s_i, st_i] = np.sum(samples[new_sample_times == sample_times[s_i, st_i]])
    new_df[:, 2::3] = new_final_data.astype(int)
    np.savetxt(alt_df_path, new_df, delimiter="\t", fmt="%d",
        header="Each row = one replicate; each set of three columns = (sampling time, total samples, derived alleles)")
    print(f"Permuted dataset {suffix} ({num_perms} replicates)")