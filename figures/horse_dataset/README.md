# ASIP and MC1R locus analyses

The scripts in this directory recreate Figure 15 of the main text, as well as the accompanying results regarding mode inference.

**Prior to running the scripts in this directory, change the current working directory to this folder.** Also, if you have not already, install the additional plotting packages required by running the command
```
pip install "emsel[plots] @ git+https://github.com/steinrue/EMSel"
```

## Figure 15A+B

Proceed via the following:

1. Create the subfolders `data`, `EM`, `output`, and `qsubs` within this subdirectory.
2. Move the files `horse_data_asip.csv` and `horse_data_mc1r.csv` from [sample_datasets/](../../sample_datasets/) to the `data` subfolder of this directory.
3. Run `python permute_target_dataset.py`.
4. Run `python SLURM_example.py`, followed by `sh meta_sim_EM.sh`. Alternatively, for each .csv file now in the `data` directory, run `emsel {data_path} {output_path} --time_after_zero --full_output -Ne 2500 --selection_modes all --progressbar`
5. Run `python horse_analysis.py`. This will generate Figure 15A+B, as well as printing the p-values for selection under each one-parameter mode. For the multiple-alternative mode inference, the log-likelihoods can be read from the .csv files output by running EMSel in step 4.
