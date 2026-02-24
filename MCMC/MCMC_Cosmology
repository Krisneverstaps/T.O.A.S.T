import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import emcee
import corner
from pathlib import Path
from scipy.integrate import quad
import scipy
from data_handler import load_snana_format, load_cov_npz, download_file


print(scipy.__version__)

# PATHS
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
hd_path   = download_file("4_DISTANCES_COVMAT/DES-Dovekie_HD.csv")
cov_path  = download_file("4_DISTANCES_COVMAT/STAT+SYS.npz")
meta_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_Metadata.csv")
inv_cov_full = load_cov_npz(cov_path)

H0_FIXED     = 70.0
N_WALKERS    = 64
N_STEPS      = 10000
BURN_IN_FRAC = 0.2
C_KM         = 299792.458

PRIORS   = {"Omega_m": (0.1, 0.6), "w": (-3.0, 0.0), "M_B": (-31.0, -18.0), "alpha": (0.0, 0.3), "beta": (0.0, 5.0)}
FIDUCIAL = [0.33, -1.0, -29.96210, 0.169, 3.14]

# LOAD DES DATA
df_hd   = load_snana_format(hd_path)
df_meta = load_snana_format(meta_path)

# MERGE HD WITH METADATA
df = df_hd[["CID", "zHD", "MUERR", "PROBIA_BEAMS"]].merge(
    df_meta[["CID", "x0", "x1", "c", "biasCor_mu", "biasCorErr_mu"]],
    on="CID", how="left"
)
df = df.dropna(subset=["zHD", "x0", "x1", "c", "biasCor_mu", "MUERR"]).reset_index(drop=True)

mask = (
    (df["zHD"] > 0) & 
    (df["PROBIA_BEAMS"] > 0.999999) & 
    (df["x0"].notna()) & 
    (df["x1"].notna()) & 
    (df["c"].notna()) &
    (df["biasCor_mu"].notna())
).values


df = df[mask].reset_index(drop=True)
inv_cov = inv_cov_full[mask][:, mask]
print(f"Data size: {len(df)}")
print(f"Matrix size: {inv_cov.shape}")

# DATA ARRAYS
z = df["zHD"].values
mu_err = df["MUERR"].values
df['total_err'] = np.sqrt(df['MUERR']**2 + df['biasCorErr_mu']**2)
yerr = df['total_err'].values


mB = -2.5 * np.log10(df["x0"].values)
df["mB"] = mB
x1 = df["x1"].values
c = df["c"].values
biasCor_mu = df["biasCor_mu"].values
GAMMA = 0.033 
print(f"Loaded {len(z)} SNe. MU = mB + alpha*x1 - beta*c + gamma/2 - biasCor_mu - M_B")

# Tripp formula
def mu_tripp(alpha, beta, M_B):
    return mB + (alpha * x1) - (beta * c) + GAMMA / 2.0 - biasCor_mu - M_B
_, _, M_B_fid, alpha_fid, beta_fid = FIDUCIAL
df["mu_obs"] = mu_tripp(alpha_fid, beta_fid, M_B_fid)
print("MU = mB + alpha*x1 - beta*c + gamma/2 - biasCor_mu - M_B")
print(df[["mB", "x1", "c", "mu_obs"]].head())

def distance_modulus(z, Omega_m, w, H0):
    Omega_de = 1.0 - Omega_m
    def E(zp):
        return np.sqrt(Omega_m * (1 + zp)**3 + Omega_de * (1 + zp)**(3 * (1 + w)))
    if np.isscalar(z):
        integral, _ = quad(lambda zp: 1.0 / E(zp), 0, z)
        d_c = C_KM * integral / H0
    else:
        d_c = np.array([C_KM * quad(lambda zp: 1.0 / E(zp), 0, zi)[0] / H0 for zi in z])
    return 5.0 * np.log10((1 + z) * d_c) + 25.0
    

def log_prior(theta):
    Omega_m, w, M_B, alpha, beta = theta
    if not (PRIORS["Omega_m"][0] < Omega_m < PRIORS["Omega_m"][1]): return -np.inf
    if not (PRIORS["w"][0] < w < PRIORS["w"][1]): return -np.inf
    if not (PRIORS["M_B"][0] < M_B < PRIORS["M_B"][1]): return -np.inf
    if not (PRIORS["alpha"][0] < alpha < PRIORS["alpha"][1]): return -np.inf
    if not (PRIORS["beta"][0] < beta < PRIORS["beta"][1]): return -np.inf
    return 0.0

def log_likelihood(theta):
    Omega_m, w, M_B, alpha, beta = theta
    mu_tripp_now = mu_tripp(alpha, beta, M_B)
    mu_theory = distance_modulus(z, Omega_m, w, H0_FIXED)
    residuals = mu_tripp_now - mu_theory
    return -0.5 * (residuals @ inv_cov @ residuals)

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    return lp + log_likelihood(theta)

n_dim = 5
theta0 = np.array(FIDUCIAL)
pos = theta0 + 1e-3 * np.random.randn(N_WALKERS, n_dim)
sampler = emcee.EnsembleSampler(N_WALKERS, n_dim, log_posterior)
sampler.run_mcmc(pos, N_STEPS, progress=True)
chains = sampler.get_chain()
print("MCMC complete")

param_names = ["Omega_m", "w", "M_B", "alpha", "beta"]
burn_in = int(BURN_IN_FRAC * N_STEPS)
flat = chains[burn_in:].reshape(-1, n_dim)
results = {}
for i, name in enumerate(param_names):
    s = flat[:, i]
    med = np.median(s)
    lo, hi = np.percentile(s, 16), np.percentile(s, 84)
    results[name] = {"median": med, "lower": med - lo, "upper": hi - med}
print("68% C.I.:")
for name in param_names:
    r = results[name]
    print(f"  {name}: {r['median']:.4f} +{r['upper']:.4f} -{r['lower']:.4f}")

# Best-fit residuals
Om       = results["Omega_m"]["median"]
w_best   = results["w"]["median"]
M_B_best = results["M_B"]["median"]
alpha_best = results["alpha"]["median"]
beta_best  = results["beta"]["median"]
mu_obs_best  = mu_tripp(alpha_best, beta_best, M_B_best)
mu_theory_z  = distance_modulus(z, Om, w_best, H0_FIXED)
residual = mu_obs_best - mu_theory_z

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]})
ax1.errorbar(z, mu_obs_best, yerr=mu_err, fmt="o", ms=2, alpha=0.3)
ax1.plot(z, mu_theory_z, "b-", lw=1, label="Best-fit theory")
ax1.set_ylabel(r"$\mu_{\rm obs}$")
ax1.legend()
ax2.scatter(z, residual, s=5, alpha=0.6)
ax2.axhline(0, color="k", ls="--")
ax2.set_xlabel("Redshift")
ax2.set_ylabel("Residual (μ_obs − μ_theory)")
plt.tight_layout()
plt.show()

fig = corner.corner(flat, labels=param_names, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.show()