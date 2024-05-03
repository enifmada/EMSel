# Recreating main text figures

This directory contains scripts to recreate the data-derived figures in the main text of Fine and Steinrücken (2024), or generate similar plots for a dataset of your choosing. Step-by-step instructions to generate each plot can be found in the README of each subfolder.

## simulation subdirectory

This subfolder contains scripts to generate the boxplots, strip plots, AUC plots, Q-Q plots, and confusion matrices found in Sections 3 and 4.2 of the manuscript. These plots do not assume that a VCF was used as input.

- `box_and_strip_plots.py` generates boxplots (e.g. Figure 4A) if `cond_only` is `False` and strip plots (e.g. Figure 6) if `cond_only` is `True`. To generate strip plots, you must first run `deltall_qqs_and_confusiontables.py`, since the Benjamini-Hochberg `classified.pkl` files are a necessary input.
- `deltall_qqs_and_confusiontables.py` generates `delta_ll` Q-Q plots (e.g Figure 4D) and confusion tables (e.g. Figure 5). If `save_bh` is `True`, this script also outputs `classified.pkl` files which can be read into `box_and_strip_plots.py` to generate strip plots.
- `qqs_and_aucs` generates chi2 Q-Q plots (e.g. Figure 4B) and AUC tables (e.g. Figure 4C). 


## gb_dataset subdirectory

This subfolder contains scripts to generate all plots found in Section 4.3 of the manuscript. These plots _do_ require a VCF as input. 

**`aggregate_data.py` must be run before the other two scripts**. This script also generates the `sample_means.txt` and `sample_missigness.txt` files needed to run data-matched simulations.

- `gb_figures.py` generates most of the figures in Section 4 - Manhattan plots (Figure 12), p-value scatterplots (Figure 13A+C, Figure 14), and multi-alternative Q-Q plots (Supplementary Figure 17). In addition, the script outputs as a pickled DataFrame for each selection mode, a set of summary statistics for each significant region. 
- `plot_trajectories.py` generates the allele frequency trajectory plots (Figure 13B+D).
