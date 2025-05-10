import numpy as np
from pathlib import Path
import pickle
from scipy.stats import chi2
from scipy.interpolate import PchipInterpolator
from emsel.emsel_util import bh_correct, get_1d_s_data_from_type
from tqdm import tqdm
import matplotlib.pyplot as plt

###### MODIFY

data_dir = "data"
EM_dir = "EM/ne/condo"
output_dir = "output"
genodata_type = "capture_only"

###### DO NOT MODIFY

Nes = np.geomspace(500,500000, 21, dtype=int)
chroms = range(1,23)

midfix = "cond"

Ne_lls = np.zeros_like(Nes, dtype=float)
condo_lls = np.zeros_like(Nes, dtype=float)
for Ne_i, Ne in enumerate(tqdm(Nes)):
    for chrom in chroms:
        unif_em_path = Path(f"{EM_dir}/GB_v54.1_{genodata_type}_c{chrom}_{midfix}_Ne{Ne}_unif_EM.pkl")
        with open(unif_em_path, "rb") as file:
            unif_hf = pickle.load(file)

        uncondo_sum = unif_hf["neutral_ll"].sum()
        condo_sum = uncondo_sum - unif_hf["cond_correction_ll"].sum()
        Ne_lls[Ne_i] += uncondo_sum
        condo_lls[Ne_i] += condo_sum


Nes_space = np.geomspace(500, 500000, 5000)
unif_spline = PchipInterpolator(Nes, Ne_lls)
unif_spline_output = unif_spline(Nes_space)
condo_spline = PchipInterpolator(Nes, condo_lls)
condo_spline_output = condo_spline(Nes_space)

print(f"spline maxes: unif {Nes_space[np.argmax(unif_spline_output)]} \n condo: {Nes_space[np.argmax(condo_spline_output)]} ")#\n icest {Nes[np.argmax(icest_Ne_lls)]}\n datameanic {Nes[np.argmax(datameanic_Ne_lls)]}")
fig, axs = plt.subplots(1,1,figsize=(5,5),layout="constrained")
axs.set_xscale("log")
axs.plot(Nes, Ne_lls, "o", label="uniform")
axs.plot(Nes, condo_lls, "o", label="cond")
axs.set_prop_cycle(None)
axs.plot(Nes_space, unif_spline_output, label="unif spline")
axs.plot(Nes_space, condo_spline_output, label="condo spline")
#axs.plot(Nes, datameanic_Ne_lls, "o", label="data-mean ic")
axs.set_prop_cycle(None)
axs.plot(Nes[np.argmax(Ne_lls)], np.max(Ne_lls), "*", ms=20)
axs.plot(Nes[np.argmax(condo_lls)], np.max(condo_lls), "*", ms=20)
axs.set_prop_cycle(None)
axs.plot(Nes_space[np.argmax(unif_spline_output)], np.max(unif_spline_output), "v", ms=20)
axs.plot(Nes_space[np.argmax(condo_spline_output)], np.max(condo_spline_output), "v", ms=20)
#axs.plot(Nes[np.argmax(icest_Ne_lls)], np.max(icest_Ne_lls), "*", ms=20)
#axs.plot(Nes[np.argmax(datameanic_Ne_lls)], np.max(datameanic_Ne_lls), "*", ms=20)

axs.legend()
axs.set_title(f"genomewide condo (condo max = {Nes_space[np.argmax(condo_spline_output)]:.0f})")
fig.savefig(Path(output_dir)/"Nes_condo_unif.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)

#fig, axs = plt.subplots(1,1,figsize=(5,5),layout="constrained")
#axs.boxplot(Ne_iters,tick_labels=Nes)
#fig.savefig(Path(output_dir)/"Nes_itercounts.pdf", format="pdf", bbox_inches="tight")
#plt.close(fig)