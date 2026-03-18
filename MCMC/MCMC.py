import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import emcee
import corner
from pathlib import Path
from data_handler import download_file, load_snana_format
from analysis_tools import calculate_physics

# PATHS
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
euclid_path = (ROOT_DIR / "data" / "Q1 euclid data.csv").resolve()
euclid_data_file_path = (ROOT_DIR / "data" / "Euclid+data.csv").resolve()

euclid_path = Path(__file__).resolve().parent.parent / "data" / "Q1 euclid data.csv"
hd_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_HD.csv")
meta_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_Metadata.csv")

# LOAD DES
df_hd = load_snana_format(hd_path).query("PROBIA_BEAMS > 0.95")
df_hd["CID_num"] = pd.to_numeric(df_hd["CID"], errors="coerce")
df_meta = load_snana_format(meta_path)[['CID', 'mB', 'x1', 'c', 'x0', 'biasCor_mu', 'biasCorErr_mu']]

# DICTIONARY FOR PROPERTIES
e_cols = {
    "DES_ID_x": "CID_num",
    #"phz_pp_mode_sfr": "SFR",
    "phz_pp_mode_sfhage": "AGE",
    "phz_pp_mode_stellarmetallicity": "METALLICITY",
    "HOST_LOGMASS": "LOGMASS"
}
df_euclid = pd.read_csv(euclid_path)[list(e_cols.keys())].rename(columns=e_cols)
df_euclid["CID_num"] = pd.to_numeric(df_euclid["CID_num"], errors="coerce")

df_ddlr = pd.read_csv(euclid_data_file_path)[['DES_ID_x', 'DDLR']]
df_ddlr = df_ddlr.rename(columns={'DES_ID_x': 'CID_num'})
df_ddlr["CID_num"] = pd.to_numeric(df_ddlr["CID_num"], errors="coerce")

# MERGE
df = (df_euclid.merge(df_hd, on="CID_num")
               .merge(df_meta, on="CID")
               .merge(df_ddlr, on="CID_num") 
               .dropna(subset=["zHD", "mB", "x1", "c", "AGE", "METALLICITY", "LOGMASS", "MUERR", "DDLR"]))


df = calculate_physics(df)

# DATA ARRAYS 
X = df[['LOGMASS', 'AGE', 'METALLICITY']].values
y = df['hubble_residual'].values
df['total_err'] = np.sqrt(df['MUERR']**2 + df['biasCorErr_mu']**2)
yerr = df['total_err'].values
splits = np.median(X, axis=0)
data = (X, y, yerr, splits)

# Sum of steps for Mass, Age, Metallicity, and SFR
def model(theta, X, splits):
    # theta: g1=MassStep, g2=AgeStep, g3=MetalStep, g4=SFRStep, T0=Offset
    g1, g2, g3, T0 = theta
    m1 = np.where(X[:, 0] >= splits[0], g1, 0) # Mass
    m2 = np.where(X[:, 1] >= splits[1], g2, 0) # Age
    m3 = np.where(X[:, 2] >= splits[2], g3, 0) # Metallicity
    return m1 + m2 + m3 + T0

# this shows how wellthe model matches the residuals
def lnlike(theta, X, y, yerr, splits):
    g1, g2, g3, T0, sig_int = theta
    
    mod = model(theta[:4], X, splits) # Use first 4 params 
    
    # Add intrinsic scatter to measurement error
    sigma2 = yerr**2 + sig_int**2
    
    return -0.5 * np.sum(((y - mod)**2 / sigma2) + np.log(2 * np.pi * sigma2))

# Priors constraining step sizes (large for no bias)
def lnprior(theta):
    g1, g2, g3, T0, sig_int = theta

    if (-0.5 < g1 < 0.5 and -0.5 < g2 < 0.5 and 
        -0.5 < g3 < 0.5 and -1.0 < T0 < 1.0 and 
        0.001 < sig_int < 0.3): # scatter
        return 0.0
    return -np.inf

# LOG-PROBABILITY
def lnprob(theta, X, y, yerr, splits):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, X, y, yerr, splits)


# Initialise MCMC
nwalkers = 64
niter = 15000
# Initial guesses,  tiny steps (0.02) and zero offset
initial = np.array([0.02, 0.02, 0.02, 0.0, 0.1])
ndim = len(initial)
p0 = [initial + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]

# MAIN 
def main(p0, nwalkers, niter, ndim, lnprob, data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 3000) # 3000 step burn-in
    sampler.reset()
    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)
    tau = sampler.get_autocorr_time()
    print(tau)
    return sampler, pos, prob, state 

new_sampler, newpos, newprob, newstate = main(p0, nwalkers, niter, ndim, lnprob, data)
new_samples = new_sampler.flatchain

# BEST FIT RESULTS
def summarize_samples(samples, labels):
    summary = []
    for i, label in enumerate(labels):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        summary.append({
            "Parameter": label,
            "Median": q50,
            "-1σ": q50 - q16,
            "+1σ": q84 - q50
        })
    return pd.DataFrame(summary)

labels_plain = ["gamma_mass", "gamma_age", "gamma_metal", "T0", "sig_int"]
summary_df = summarize_samples(new_samples, labels_plain)

print("\nMCMC Parameter Summary:")
print(summary_df.to_string(index=False))

# VISUALIZATION 
truths = np.median(new_samples, axis=0)

fig = corner.corner(
    new_samples,
    labels=[r'$\gamma_\mathrm{Mass}$',
            r'$\gamma_\mathrm{Age}$',
            r'$\gamma_\mathrm{Metal}$',
            r'$T_0$',
            r'$\sigma_\mathrm{int}$'],
    truths=truths, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3f", 
    title_kwargs={"fontsize": 12}, label_kwargs={"fontsize": 14}, smooth=1.0
)


print("\nPhysical Interpretation:")
for i, row in summary_df.iterrows():
    sig = abs(row["Median"]) / np.mean([row["-1σ"], row["+1σ"]])
    print(f"{row['Parameter']:>12}: {row['Median']:+.3f} mag "
          f"({sig:.2f}σ)")


out_dir = ROOT_DIR / "outputs"
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "MCMC_Corner_PlotNEW.png", dpi=150)
plt.show()

