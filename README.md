# EMSel
Code accompanying Fine and Steinrücken (2024). We provide the ability to simulate (`simulate_data.py`) and analyze (`run_emsel.py`) a time-series allele frequency dataset under multiple modes of selection (additive, dominant, recessive, over/underdominance, general diploid), as well as the data and code to reproduce the figures from our paper.

## Running EMSel

To run EMSel (via `run_emsel.py`), you must have either a CSV file or a VCF file.

### Using EMSel with CSVs

The CSV should be formatted as the following:
- each locus/replicate should be in its own row.
- each row should contain 3N non-zero values, consisting of N samples each formatted as the triple (sampling time, number of derived alleles, number of total samples).
- samples of size zero are allowed, and are interpreted identically to a gap between sampling times.
- rows do not need to have the same length.
- sampling times can be expressed in years or generations: see the use of the -ytg flag below.

An example of a properly-formatted CSV is available in the `simulations` folder.

### Using EMSEL with VCFs

Fortunately the formatting of VCFs is standardized. If the VCF can be read by `scikit-allele`, it can be used with EMSel. In addition, however, using EMSel with a VCF requires a file containing the same strings as the `samples` key in VCF and a corresponding sampling time (in years or generations, see the -ytg flag below) for each sample.

### Minimal example and output

A minimal sample call to EMSel with a CSV:
`python run_emsel.py input_data.csv output_EM.pkl.bz2`

A minimal sample call with a VCF:
`python run_emsel.py input_data.vcf output_EM.pkl.bz2 --sample_file individuals.table --sample_cols Genetic_ID Date_mean`

Both of these will create the file `output_EM.pkl.bz2` containing the results of running EMSel in all available modes of selection. The output is a nested dictionary-of-dictionaries containing the following keys, letting `N` be the number of replicates in the dataset:
- `neutral_ll` (N,) - the log-likelihood for each replicate calculated under neutrality (s1 = s2 = 0).
- `neutral_ic` (N, varies) - the estimated initial distribution parameters for each replicate calculated under neutrality. The second dimension depends on which initial distribution is used for calculation.
- `neutral_itercount` (N,) - the number of iterations for convergence for each replicate under neutrality.
- for each mode of selection analyzed under, a sub-dictionary with key `{update_type}_run` (e.g. `add_run` for additive selection), containing the following keys:
  - `s_final` (N, 2) - the maximum-likelihood estimate of the selection coefficients for each replicate.
  - `ll_final` (N,) - the maximum log-likelihood estimate for each replicate.
  - `ic_dist` (N, varies) - the estimated initial distribution parameters for each replicate. The second dimension depends on which initial distribution is used for calculation.
  - `itercount_hist` - the number of iterations for convergence for each replicate.
  - `exit_codes` - exit codes indicating the termination statuts of each replicate. See section "Exit Codes".
 
### Command-line arguments

In addition to the required `input` and `output` paths, EMSel has the following optional arguments:

```
-ytg, --years_to_gen <float, default=1>
Number of years per generation, used to convert a VCF or CSV to generations. If the sampling times in the sample file or the CSV are in generations, use the defualt of 1. Note that for the --save_csv flag, the CSV output will be in generations.
```


```
--s_init <2 floats, default=0 0>
(s1, s2) for initialization of the HMM and the inital "guess" for each replicate.```
```

```
-ic, --init_cond <str, default='uniform'>
Initial condition (pi_0 in HMM language) for initialization of the HMM. Options are: 
- "uniform" - uniform, equiprobable prior (recommended/default)
- "delta" - use the "--ic_dict p x", where 0 <= x <= 1 to set the initial condition to a delta function at p = x.
- "beta" - use "--ic_dict beta_coef alpha", where alpha is a real number, to set the initial condition to a symmetric beta function beta(alpha, alpha).
- "spikeandslab" - use "--ic_dict spike_frac x spike_loc y", where 0 <= x, y <= 1 to set the initial condition to a mixture of uniform with weight 1-x and a delta function at p = y with weight x.
- "theta" - generates an initial condition where the probability in hidden state i is proportional to 1/i. Scales with the inputted value of -mu.
- "theta-trunc" - use "--ic_dict p x", where 0 <= x <= 1 to set the initial condition to one in which the probaiblity in hidden state i is proportional to 1/i, but states with allele frequency <= x have weight 0 (intended to prevent the weights from diverging near zero).
```

```
--ic_dict <1+ arguments of the form 'str int'>
Additional arguments if -ic is not "uniform" or "theta".
```

```
-Ne <int, default=10000>
Effective population size. Only used in the calculation of the transition state matrix.
```

```
-mu <float, default=1.25e-8>
Mutation rate, expressed mutations per base pair per generation. Only affects the weights in the "theta" and "theta-trunc" initial distributions.
```


```
--hidden_interp <str, default='chebyshev'>
Whether to use Chebyshev nodes for spacing of hidden states (highly recommended) or linear (via --hidden_interp linear). Chebyshev nodes do not impact runtime and appear to significantly improve accuracy of selection coefficient estimation at high selection coefficients, especially for certain modes of selection.
```

```
--ic_update_type <str, default='beta'>
Method of estimating the initial condition. Options are:
- "beta" - estimate the parameters of a beta distribution. Output dictionary values involving the initial distribution will have shape (N,2)
- "delta" - estimate the parameter of a delta distribution. Output dictionary values will have shape (N, 1)
- "baumwelch" - use the standard Baum-Welch EM update rules to estimate the weights for all hidden states. Output dictionary values will have shape (N, Ns), where Ns = the number of hidden states.
- any other string will cause the initial condition to not be estimated (i.e. pi_k = pi_0 for all iterations k). Output dictionary values will have shape (N, 1).
```

```
--update_types <1+ str, default='all'>
Which modes of selection to analyze under. You can list as many of ["neutral", "add", "dom", "rec", "het", "full"] as you would like. Neutral is automatically run. The default, "all", is a shorthand for running all modes of selection.
```

```
-nc, --num_cores <int, default=1>
Number of cores to parallelize over. Joblib is used for parallelization, making EMSel easily parallizable on a computing cluster.
```

```
-ns, --num_states <int, default=500>
Number of hidden states to use in the HMM. Computing time scales as O(hidden states^3), roughly. Accuracy is reduced below approximately 200-250 hidden states.
```

```
-t, --tol <float, default=1e-3>
Stopping criterion - EMSel is said to have converged if log-likelihood at iteration k+1 - log-likelihood at iteration k < tol.
```

```
-m, --maxiter <int, default=2000>
Maximum number of iterations of EMSel before terminating. Across both our simulated datasets and the UK aDNA dataset, the maximum number of iterations seen was [idk]
```

```
--min_itercount <int, default=5>
Minimum number of iterations for EMSel to run, even if the stopping criterion is met earlier. Helps alleviate strangeness in the distribution of log-likelihoods/p-values near 1. However, since p-values near 1 are typically unimportant, setting this to 0 to slightly speed up computation is reasonable.
```

```
--sample_file <str>
When input is a VCF, path to a sample file readable by pandas.read_csv containing a column of IDs matching the IDs in the VCF as well as a column of sampling times (in years or generations, use -ytg to normalize to generations if years).
```

```
--sample_cols <2 strs>
Column names, in (IDs, times) order, to extract from the sample file.
```

```
--save_csv
If used and input is a VCF, saves a CSV of the same name containing the intermediate conversion of the VCF into (sampling time, derive alleles, total sample) triplets to speed up future runs. Note that the saved CSV will include conversion from years to generations. If the input is a VCF and a CSV of the same name exists, the CSV will be used with years-to-generation=1.
```

```
--progressbar
Track the progress of EMSel with a tqdm progressbar.
```

## Simulating data

`simulate_data.py` uses the discrete Wright-Fisher model to simulate allele frequencies under a given selection scenario (selection strength, mode of selection, initial condition, number of generations simulated, sampling scheme). 

### Command-line arguments

There is one required argument - a path to an output directory. In addition, there are the following positional arguments:

```
-n, --num_sims <int, default=100>
Number of replicates simulated from each set of simulation parameters.
```

```
-s, --sel_coeffs <float or sequence of floats, default=.005, .01, .025, .05>
Selection coefficients to simulate. For additive selection, this value is s_2 (the fitness of the derived homozygote), with s_1 = s_2 / 2. All other modes of selection are governed by a single parameter. Simulating under neutrality should be done by adding "neutral" to the --sel_types flag, not by including 0 in this flag.
```

```
--sel_types <str or sequence of strs, default=neutral add dom rec over under>
Modes of selection to simulate under. Must be a (non-strict) subset of {"neutral", "add", "dom", "rec", "over", "under"}.
```

```
-g, --num_gens <int or sequence of ints, default=101 251 1001>
Number of generations to simulate for. 
```

```
-ic, --init_conds <float, str, or sequence of floats and strs, default=.005, .25, "recip">
Initial conditions to simulate under. Floats are interpreted as initializing each simulation from the same fixed allele frequency. Currently, the only non-float option is "recip", which samples from a distribution where the probability of initial frequency p is proportional to 1/p. TODO: handle strings other than "recip" properly and describe their initial distributions.
```

```
-ns, --num_samples <int, default=50>
Number of haploid samples drawn at each sampling timepoint.
```

```
-st, --sampling_times <int, default=11>
Number of equally-spaced timepoints to sample at. Samples are taken at the first generation, the last generation and (sampling_times - 2) points equally spaced between.
```

```
-Ne <int, default=10000>
Effective population size for the simulations.
```

```
--data_matched <2 strs>
Provide the path to a sim_data file and a sample_sizes.table file to override the -ic, -g, -ns, and -st flags and simulate under the same initial distribution, number of generations, and sampling scheme as the inputted real data file. The sim_data file can be obtained by running EMSel, the sample_sizes.table file by running the AADR_to_VCF pipeline.
```

```
--small_s
Use of this flag indicates that the formula for updating allele frequencies in the Wright-Fisher model should use the small s approximation (p' = p + p(1-p)*((1-2p)s1 + p*s2)) rather than the full formula, p' = p(1+s1(1-p)+s2*p)/(1+2*s1*p+s2*p^2-2s1*p**2).
```

```
--seed <int>
Seed to simulate from. Simulations from identical seed values are identical.
```

```
--save_plots
If used, one plot per simulation condition plotting the true allele frequency for each replicate as well as the mean allele frequency will be produced.
```

```
--suffix <str>
Adds a suffix to each file to help distinguish different simulation runs.
```

When specifying multiple values for sel_coeffs, sel_types, num_gens, and init_conds, simulation is done for each combination on values (i.e. on itertools.product(sel_coeffs, sel_types, num_gens, init_conds), for a total of len(sel_coeffs)*len(sel_types)*len(num_gens)*len(init_conds) sets of files.

### Output and examples

`simulate_data.py` outputs three different file types:

1. `args_{suffix}.pkl` - contains a dictionary of the parameters used to run all sets of simulations. 

For each simulation condition, the following two files are outputted:

2. `{exp_name}_{suffix}data.csv` - contains the sampled allele frequencies and sampling times (see Using EMSel with CSVs for full formatting description).
3. `{exp_name}_{suffix}pd.bz2` - a bz2-zipped pickle file containing the simulation parameters for this particular simulation condition.

Sample calls to `simulate_data.py` for a non-data-matched set of simulations and a data-matched simulation are as follows:

`python simulate_data.py simulations_folder -s .01 .1 -g 101 251 -ic .05 recip --suffix big_s`

`python simulate_data.py data_simulations -s .005 .05 .2 --sel_types neutral add rec --data_matched UK_aDNA_sim_data.pkl UK_aDNA_sample_sizes.table`
