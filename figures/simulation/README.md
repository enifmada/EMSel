<!--- emsel-sim data/real_matched -s .005 --sel_types neutral --seed 5 -n 10000 --data_matched data/real_matched/neutral_g125_dal_special_means.txt data/GB_v54.1_capture_only_missingness.txt data/GB_v54.1_capture_only_sample_sizes.table --suffix resim -Ne 9987 --->

# Simulation studies

The scripts in this directory can be used to recreate all figures in Sections 3 (Figures 4-8) and 4.2 (Figures 9-11) of the manuscript, as well as figures S.1-S.17 and Figure S.29A in the Supplemental Material. This README is organized by what simulations need to be run to produce a given set of figures, with the analysis commands and scripts to produce the figures noted therein.

**Prior to running the scripts in this directory, change the current working directory to this folder.** Also, if you have not already, install the additional plotting packages required by running the command
```
pip install "emsel[plots] @ git+https://github.com/steinrue/EMSel"
```
## Directory setup

Run `python make_directories.py` in this directory to set up the correct folder structure. Alternatively, create the following directories and subdirectories relative to this directory:
- `data`, `data/pure_sim`, `data/pure_sim/boxplots`, `data/pure_sim/param_variation`, `data/ibdne`, `data/ibdne/boxplots`, `data/ibdne/big`, `data/real_matched`, `data/real_matched/boxplots`, `data/real_matched/permutations`
- `EM`, `EM/pure_sim`, `EM/pure_sim/boxplots`, `EM/pure_sim/param_variation`, `EM/ibdne`, `EM/ibdne/boxplots`, `EM/ibdne/big`, `EM/real_matched`, `EM/real_matched/boxplots`, `EM/real_matched/permutations`
- `output`, `output/pure_sim`, `output/ibdne`, `output/real_matched`
- `classified`, `qsubs`

## Figures 3-5, 7, S.1-S.10

These figures are all generated from the large simulation described in Section 3. The simulated data can be recreated and reanalyzed by taking the following steps:

1. Run `emsel-sim data/pure_sim -s .005 .01 .025 .05 -g 101 251 1001 -ic recip .005 .25 --sel_types add dom rec over under --seed 5 -n 1000` to generate the non-neutral simulations.
2. Run `emsel-sim data/pure_sim -s .005 -g 101 251 1001 -ic recip .005 .25 --sel_types neutral --seed 5 -n 10000` to generate the neutral simulations.
3. Run `emsel-sim data/pure_sim -s .005 -g 101 251 1001 -ic recip .005 .25 --sel_types neutral --seed 9734 -n 10000` to generate the neutral re-simulations needed for figure 3D.
4. Run `python SLURM_example.py` with `EM_dir = Path('EM/pure_sim')` and `data_dir = Path('data/pure_sim')` at the beginning of the script, modifying the other parameters to your liking. This should generate 186 files to be run. Then, run `sh meta_sim_EM.sh` to submit the jobs to the cluster. Alternatively, for each dataset in `data`, running `emsel data/pure_sim/{file_name}_data.csv EM/pure_sim/{file_name}_EM --time_after_zero -maf 0 --min_sample_density 0 --full_output`.

Once the EM files are generated, the figures can be generated by running the following scripts in the specified configuration:

### Figures 3A, S.1-8A:
Set the following parameters at the beginning of the script `box_and_strip_plots.py` and run it using `python box_and_strip_plots.py`:
```
sel_strs = [.005, .01, .025, .05]
num_gens_list = [101, 251, 1001]
init_dists = [.005, .25, "recip"]
cond_only = False

data_dir = "data/pure_sim"
EM_dir = "EM/pure_sim"
output_dir = "output/pure_sim"
classified_dir = "classified"
```

### Figures 3B+C, S.1-8B+C:
Set the following parameters at the beginning of the script `qqs_and_aucs.py` and run it using `python qqs_and_aucs.py`:
```
sel_strs = [.005, .01, .025, .05]
num_gens_list = [101, 251, 1001]
init_dists = [.005, .25, "recip"]

EM_dir = "EM/pure_sim"
output_dir = "output/pure_sim"
```

### Figures 3D, 4, S.1-8D:
Set the following parameters at the beginning of the script `d2_qqs_and_confusiontables.py` and run it using `python d2_qqs_and_confusiontables.py`:
```
sel_strs = [.005, .01, .025, .05]
num_gens_list = [101, 251, 1001]
init_dists = [.005, .25, "recip"]

EM_dir = "EM/pure_sim"
output_dir = "output/pure_sim"
classified_dir = "classified"
```

### Figure 5:
Set the parameter `cond_only = True` at the beginning of the script `box_and_strip_plots.py` and leave the rest of the configuration the same as Figures 3A et al. above. Run it using `python box_and_strip_plots.py`. Note that this will only run successfully after `d2_qqs_and_confusiontables.py` has been run, since the `_classified.pkl` files must be generated.

### Figures 7A+B:

Set the following parameters at the beginning of the script `mismatched_boxplots.py` and run it using `python mismatched_boxplots.py`:
```
data_dir = "data/pure_sim"
EM_dir = "EM/pure_sim"
output_dir = "output"
```

### Figures 7C+D:
Run `python mismatched_auc_plots.py`. Modify the directory paths in the script if needed.

### Figure S.9:
Run `python unconstrained_scatterplot.py`. Modify directory paths as needed.

### Figure S.10:
First, run `python SLURM_extra_sims.py` (this should generate 63 files to be run) followed by `sh meta_sim_EM.sh`. This will generate the EM files needed for figures S.12.  Alternatively, for every file in `data` that contains `g251_d25`, run `emsel data/pure_sim/{file_name}_g251_d25_data.csv EM/{file_name}_g251_d25_ns100_linear_EM --time_after_zero -maf 0 --min_sample_density 0 --full_output -hs 100 --hidden_interp linear --ic_update_type fixed`, `emsel data/pure_sim/{file_name}_g251_d25_data.csv EM/{file_name}_g251_d25_linear_EM --time_after_zero -maf 0 --min_sample_density 0 --full_output --hidden_interp linear --ic_update_type fixed` and `emsel data/pure_sim/{file_name}_g251_d25_data.csv EM/{file_name}_g251_d25_fixed_ic_EM --time_after_zero -maf 0 --min_sample_density 0 --full_output -ic_update_type fixed` to reanalyze the 251 generations, init freq = 0.25 data under the (100 linearly interpolated hidden states, 500 linearly interpolated hidden states, 500 chebyshev interpolated hidden states, and 500 chebyshev interpolated hidden states but no initial condition estimation) conditions, respectively.

Then, to generate Figure S.10, set the following parameters at the beginning of the script `file_str_boxplots.py` and run it using `python file_str_boxplots.py`:
```
sel_strs = [.005, .01, .025, .05]
num_gens = 251
init_dist = .25

data_dir = "data/pure_sim"
EM_dir = "EM/pure_sim"
output_dir = "output/pure_sim"

file_strs = ["ns100_linear_", "linear_", "fixed_ic_", ""]
blank_name_str = "standard_
```

## Figure 6:
This figure requires additional simulations. Proceed via the following:

1. For ns in `[6, 20, 50, 100, 200]`, run `emsel-sim data/pure_sim/param_variation -s .025 -g 251 -ic .25 --sel_types add dom rec --seed 5 -n 1000 -ns {ns}`.
2. For st in `[2, 5, 11, 35, 101]`, run `emsel-sim data/pure_sim/param_variation -s .025 -g 251 -ic .25 --sel_types add dom rec --seed 5 -n 1000 -st {st}`.
3. For Ne in `[100, 1000, 10000, 100000, 1000000]`, run `emsel-sim data/pure_sim/param_variation -s .025 -g 251 -ic .25 --sel_types add dom rec --seed 5 -n 1000 -Ne {Ne}`.
4. Run `python SLURM_example.py` with `EM_dir = Path('EM/pure_sim/param_variation')` and `data_dir = Path('data/pure_sim/param_variation')` at the beginning of the script. This should generate 60 files to be run. Then, submit the jobs using `sh meta_sim_EM.sh`. Alternatively, for all files now in `data/pure_sim/param_variation`, run `emsel data/pure_sim/param_variation/{file_name}_data.csv EM/{file_name}_EM --time_after_zero -maf 0 --min_sample_density 0 --full_output`
5. If you did not run the SLURM script, for all combinations of `sel_type` in `[add, dom, rec]` and `hs` in `[100, 250, 500, 1000, 2000]`, run `emsel data/pure_sim/{sel_type}_s025_g251_d25_data.csv EM/param_variation/{sel_type}_s025_g251_d25_hs{hs}_EM --time_after_zero -maf 0 --min_sample_density 0 -hs hs --selection_modes {sel_type} --full_output`.
6. Run `python param_variation_plots.py`.

## Figure 8:
This figure also requires additional simulations. Proceed via the following:

1. Run `python SLURM_Ne_pure_sim.py` followed by `sh meta_sim_EM.sh` to simulate the additional datasets needed to generate the Ne boxplots. Or, for all combinations of `Ne` in `[2500, 10000, 40000]` and `seed` in `[100, 101, ..., 124]`, run `emsel-sim data/pure_sim/boxplots -s .005 --sel_types neutral --seed {seed} -n 10000 --save_plots --suffix seed{seed} -g 251 -ic .25 -Ne {Ne}`
2. Set the following parameters at the beginning of the script `SLURM_Ne_computation.py` and run it with the commands `python SLURM_Ne_computation.py` followed by `sh meta_sim_EM.sh`.
```
EM_dir = Path('EM/pure_sim/boxplots')
data_dir = Path('data/pure_sim/boxplots')
```

3. Set the following parameters at the beginning of the script `ne_condo_boxplots.py` and run it with the command `python ne_condo_boxplots.py`.
```
data_dir = "data/pure_sim/boxplots"
EM_dir = "EM/pure_sim/boxplots"
output_dir = "output/pure_sim"

sim_Nes = [2500, 10000, 40000]
prefixes = [*[f"g251_d25_Ne{sim_Ne}" for sim_Ne in sim_Nes]]
labels = [rf"$N_e = {{{str(i)}}}$" for i in sim_Nes]
EM_dirs = ["EM/pure_sim/boxplots", "EM/pure_sim/boxplots", "EM/pure_sim/boxplots"]
```


## Figures 9-11:
1. Run the pipeline in the [figures/gb_dataset/](../gb_dataset/) folder up to the point where the `GB_v54.1_capture_only_means.txt` and `GB_v54.1_capture_only_missingness.txt` files are created (i.e. run the "All figures" section), then copy these two files and the `GB_v54.1_capture_only_sample_sizes.table` file into `data`. Additionally, copy the files `ibdne_raw.txt` and `ibdne_original.txt` from [sample_datasets](../../sample_datasets/)_to the `data` folder.
2. Run `emsel-sim data/ibdne -s .005 .01 .025 .05 --sel_types neutral add dom rec over under --seed 9734 -n 10000 --data_matched data/GB_v54.1_capture_only_means.txt data/GB_v54.1_capture_only_missingness.txt data/GB_v54.1_capture_only_sample_sizes.table --vary_Ne data/ibdne_raw.txt`. This should generate 21 files, each with "g125_dal_special" somewhere in their name.
3. Run `python SLURM_Ne_computation.py` with `EM_dir = Path('EM/ibdne')` and `data_dir = Path('data/ibdne')` at the beginning of the script. This will generate 21 files to be run. Then, run `sh meta_sim_EM.sh` to submit the jobs to the cluster.
4. Run `python interpolate_Ne_grid.py` with `EM_dir = "EM/ibdne"` at the beginning of the script. This should output `Estimated Ne: 35313`. If it does not, modify the line `Ne = 35313` in `SLURM_example.py` and `plot_data_matched_permutations.py` to the correct value, as well as the the line `EM_suffix = "Ne35313_"` in `box_and_strip_plots.py`, `d2_qqs_and_confusiontables.py`, and `qqs_and_aucs.py`.
5. Run `python SLURM_example.py` with `EM_dir = Path('EM/ibdne')` and `data_dir = Path('data/ibdne')` at the beginning of the script. This should generate 105 files to be run. Then, run `sh meta_sim_EM.sh` to submit the jobs to the cluster. Alternatively, for each file created, run `emsel data/ibdne/{file_name}_data.csv EM/{file_name}_EM --time_after_zero --full_output -Ne 35313`.

If you ran the SLURM script, run `python combine_split_runs.py`. You should have 21 files.

Then:

### Figure 9A:

Set the following parameters at the beginning of the script `box_and_strip_plots.py` and run it using `python box_and_strip_plots.py`:
```
sel_strs = [.005, .01, .025, .05]
num_gens_list = [125]
init_dists = ["real_special"]
cond_only = False

data_dir = "data/ibdne"
EM_dir = "EM/ibdne"
output_dir = "output/ibdne"
classified_dir = "classified"
```

### Figure 9B+C:
Set the following parameters at the beginning of the script `qqs_and_aucs.py` and run it using `python qqs_and_aucs.py`:
```
sel_strs = [.005, .01, .025, .05]
num_gens_list = [125]
init_dists = ["real_special"]

EM_dir = "EM/ibdne"
output_dir = "output/ibdne"
```

### Figure 9D+10:
Set the following parameters at the beginning of the script `d2_qqs_and_confusiontables.py` and run it using `python d2_qqs_and_confusiontables.py`:
```
sel_strs = [.005, .01, .025, .05]
num_gens_list = [125]
init_dists = ["real_special"]

EM_dir = "EM/ibdne"
output_dir = "output/ibdne"
classified_dir = "classified"
```

### Figure 11:
Run `python box_and_strip_plots.py` with the same configuration as Figure 9A, but with `cond_only = True`.


## Figures S.11-S.13:
Run `emsel-sim data/real_matched -s .005 .01 .025 .05 --sel_types neutral add dom rec over under --seed 5 -n 10000 --data_matched data/GB_v54.1_capture_only_means.txt data/GB_v54.1_capture_only_missingness.txt data/GB_v54.1_capture_only_sample_sizes.table -Ne 9715` to generate the data-matched simulations. Then, repeat all steps used to generate Figures 9-11 above, replacing `"ibdne"` with `"real_matched"` everywhere it appears (i.e. setting `EM_dir = "EM/real_matched"`, etc.). Additionally, the Ne estimated should be 10496 rather than 35313.

## Figure S.14:
First, run `emsel-sim data/ibdne/big -s .005 --sel_types neutral --seed 90014521 -n 10000 --data_matched data/GB_v54.1_capture_only_means.txt data/GB_v54.1_capture_only_missingness.txt data/GB_v54.1_capture_only_sample_sizes.table --vary_Ne data/ibdne_raw.txt --suffix fit --save_plots` and `emsel-sim data/ibdne/big -s .005 --sel_types neutral --seed [s_i] -n 100000 --data_matched data/GB_v54.1_capture_only_means.txt data/GB_v54.1_capture_only_missingness.txt data/GB_v54.1_capture_only_sample_sizes.table --vary_Ne data/ibdne_raw.txt --suffix plot[i] --save_plots` , with i = [1,2,3] and s_i = [2653305, 59032744, 82459611] to generate the 10k fitting replicates and 300k plotting replicates, matched to the IBDNe dataset, for this figure. 

Next, run `python SLURM_example.py`, followed by `python combine_split_runs.py`, changing the data directory to `data/ibdne/big` and the EM directory to `EM/ibdne/big` as appropriate. Then, run `python big_bootstrap_fitting.py` to fit the generalized gamma and chi-squared distributions needed for the plot. Lastly, run `python big_d2_qqs.py` to generate the Q-Q plot comparing the fits of the chi-squared(1), chi-squared(2), chi-squared(k), and generalized gamma distributions.

## Figure S.15:
Run `python plot_ibdne_trajectory.py`.

## Figure S.16:
This figure requires additional simulations. Proceed via the following:

1. Run `python SLURM_Ne_others_sim.py`, followed by `sh meta_sim_EM.sh`. This will simulate the additional datasets needed to generate the Ne boxplots.
2. Set the following parameters at the beginning of the script `SLURM_Ne_computation.py` and run it with the commands `python SLURM_Ne_computation.py` followed by `sh meta_sim_EM.sh`.
```
EM_dir = Path('EM/ibdne/boxplots')
data_dir = Path('data/ibdne/boxplots')
```
Repeat with the following parameters:
```
EM_dir = Path('EM/real_matched/boxplots')
data_dir = Path('data/real_matched/boxplots')
```

3. Set the following parameters at the beginning of the script `ne_condo_boxplots.py` and run it with the command `python ne_condo_boxplots.py`.
```
data_dir = "data/ibdne/boxplots"
EM_dir = "EM/ibdne/boxplots"
output_dir = "output/ibdne"

prefixes = ["g125_dal_special_realmatch", "g125_dal_special_ibdne"]
labels = ["Real matched", "IBDNe"]
EM_dirs = ["EM/real_matched/boxplots", "EM/ibdne/boxplots"]
```

## Figure S.17:
This figure requires the simulations used to generate Figure S.16 to be completed as well. Then, run `python flat_ne_plots.py` to generate this figure.

## Figure S.29A:
1. Run `python permute_data_matched.py`.
2. Run `python SLURM_example.py` with `EM_dir = Path('EM/real_matched/permutations')` and `data_dir = Path('data/real_matched/permutations')` at the beginning of the script. This should generate 100 files to be run. Then, run `sh meta_sim_EM.sh`. Alternatively, for i in range(100), run `emsel data/real_matched/permutations/neutral_g125_dal_special_perm{i}_data.csv EM/real_matched/permutations/neutral_g125_dal_special_perm{i}_EM --time_after_zero -maf 0 --min_sample_density 0 --full_output`.
3. If you ran the SLURM script, run `python combine_split_runs.py` with `EM_dir = "EM/real_matched/permutations"` at the top. This will reformat the files (but not actually combine anything). You should have 100 files. 
4. Run `python plot_data_matched_permutations.py`.
