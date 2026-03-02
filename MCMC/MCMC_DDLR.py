import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import emcee
import corner
from pathlib import Path
from data_handler import download_file, load_snana_format
from analysis_tools import calculate_physics

# SETUP PATHS
ROOT_DIR = Path(__file__).resolve().parent.parent
data_dir = ROOT_DIR / "data"

euclid_cols = [
    "DES_ID_x", 
    "HOST_LOGMASS", 
    "phz_pp_mode_sfhage", 
    "phz_pp_mode_stellarmetallicity"
]

df_props = pd.read_csv(data_dir / "Q1 euclid data.csv", usecols=euclid_cols)
df_morph = pd.read_csv(data_dir / "Euclid+data.csv", usecols=["DES_ID_x", "DDLR"], thousands=',')


df_props = df_props.rename(columns={
    "phz_pp_mode_sfhage": "AGE",
    "phz_pp_mode_stellarmetallicity": "METALLICITY",
    "HOST_LOGMASS": "LOGMASS"
})


for d in [df_morph, df_props]:
    d["DES_ID_x"] = pd.to_numeric(d["DES_ID_x"].astype(str).str.replace('"', '').str.strip(), errors="coerce")

# Combine Euclid Data
euclid_all = df_morph.merge(df_props, on="DES_ID_x", how="inner").dropna()


df_hd = load_snana_format(download_file("4_DISTANCES_COVMAT/DES-Dovekie_HD.csv"))
df_meta = load_snana_format(download_file("4_DISTANCES_COVMAT/DES-Dovekie_Metadata.csv"))

df_hd["CID_num"] = pd.to_numeric(df_hd["CID"], errors="coerce")
df_hd = df_hd[df_hd["PROBIA_BEAMS"] > 0.95]


df = (
    euclid_all
    .merge(df_hd[["CID_num", "CID", "zHD", "MU", "MUERR"]], left_on="DES_ID_x", right_on="CID_num")
    .merge(df_meta[["CID", "mB", "x1", "c", "x0", "biasCor_mu", "biasCorErr_mu"]], on="CID")
    .dropna(subset=["zHD", "LOGMASS", "AGE", "METALLICITY", "DDLR"])
)

df = calculate_physics(df)

# DATA ARRAYS 
X = df[['LOGMASS', 'AGE', 'METALLICITY']].values
y = df['hubble_residual'].values
ddlr = df['DDLR'].values
df['total_err'] = np.sqrt(df['MUERR']**2 + df['biasCorErr_mu']**2)
yerr = df['total_err'].values
splits = np.median(X, axis=0)
data = (X, y, yerr, ddlr, splits)

# Sum of steps for Mass, Age, Metallicity, and SFR
def model(theta, X, ddlr, splits):
    # theta: g1=MassStep, g2=AgeStep, g3=MetalStep, g4=SFRStep, T0=Offset
    g1, g2, g3, T0, sig = theta
    fading = np.exp(-0.5 * (ddlr/sig)**2)
    m1 = np.where(X[:, 0] >= splits[0], g1*fading, 0) # Mass
    m2 = np.where(X[:, 1] >= splits[1], g2*fading, 0) # Age
    m3 = np.where(X[:, 2] >= splits[2], g3*fading, 0) # Metallicity
    return m1 + m2 + m3 + T0

# this shows how wellthe model matches the residuals
def lnlike(theta, X, y, yerr, ddlr, splits):
    mod = model(theta, X, ddlr, splits)
    return -0.5 * np.sum(((y - mod) / yerr)**2)

# Priors constraining step sizes (large for no bias)
def lnprior(theta):
    g1, g2, g3, T0, sig = theta
    if (-0.5 < g1 < 0.5 and -0.5 < g2 < 0.5 and 
        -0.5 < g3 < 0.5  and -1.0 < T0 < 1.0 and 0.00001 < sig < 10):
        return 0.0
    return -np.inf

# LOG-PROBABILITY
def lnprob(theta, X, y, yerr, ddlr, splits):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, X, y, yerr, ddlr, splits)


# Initialise MCMC
nwalkers = 64
niter = 50000
# Initial guesses,  tiny steps (0.02) and zero offset
initial = np.array([0.02, 0.02, 0.02, 0.0, 1.5])
ndim = len(initial)
p0 = [initial + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]

# MAIN 
def main(p0, nwalkers, niter, ndim, lnprob, data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 5000) # 5000 step burn-in
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

labels_plain = ["gamma_mass", "gamma_age", "gamma_metal", "T0"]
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
            r'$\sigma_\mathrm{decay}$'],
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
plt.savefig(out_dir / "MCMC_Corner_Plot_DDLR.png", dpi=150)
plt.show()
