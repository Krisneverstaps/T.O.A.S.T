import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import emcee
import corner
from pathlib import Path
from data_handler import download_file, load_snana_format
from analysis_tools import calculate_physics

ROOT_DIR = Path(__file__).resolve().parent.parent
data_dir = ROOT_DIR / "data"

euclid_cols = ["DES_ID_x", "HOST_LOGMASS", "phz_pp_mode_sfhage", "phz_pp_mode_stellarmetallicity"]
df_props = pd.read_csv(data_dir / "Q1 euclid data.csv", usecols=euclid_cols)
df_morph = pd.read_csv(data_dir / "Euclid+data.csv", usecols=["DES_ID_x", "DDLR"], thousands=',')

df_props = df_props.rename(columns={
    "phz_pp_mode_sfhage": "AGE",
    "phz_pp_mode_stellarmetallicity": "METALLICITY",
    "HOST_LOGMASS": "LOGMASS"
})

for d in [df_morph, df_props]:
    d["DES_ID_x"] = pd.to_numeric(d["DES_ID_x"].astype(str).str.replace('"', '').str.strip(), errors="coerce")

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
df = df[(df['DDLR'] < 4) & (df['DDLR'] > -2)]
df = calculate_physics(df)

# X_raw for split calculation, X for standardized model input
X_raw = df[['LOGMASS', 'AGE', 'METALLICITY']].values
splits = np.median(X_raw, axis=0)

X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X = (X_raw - X_mean) / X_std

y = df['hubble_residual'].values
ddlr = df['DDLR'].values
df['total_err'] = np.sqrt(df['MUERR']**2 + df['biasCorErr_mu']**2)
yerr = df['total_err'].values

#MODEL

X_center = X_raw - np.median(X_raw, axis=0)

def model(theta, X_center, ddlr):
    g_mass, g_age, g_met, T0, sig_decay = theta

    fading = np.exp(-0.5 * (ddlr / sig_decay)**2)

    host_term = (
        g_mass * X_center[:,0] +
        g_age  * X_center[:,1] +
        g_met  * X_center[:,2]
    )

    return fading * host_term + T0

#Likelihood
def lnlike(theta, X_center, y, yerr, ddlr):
    g1, g2, g3, T0, sig_decay, sig_int = theta

    mod = model(theta[:5], X_center, ddlr)

    sigma2 = yerr**2 + sig_int**2

    return -0.5 * np.sum(((y - mod)**2 / sigma2) + np.log(2*np.pi*sigma2))


#priors
def lnprior(theta):
    g1, g2, g3, T0, sig_decay, sig_int = theta
    
    # Gaussian prior on T0 (centered at 0)
    lp = -0.5 * (T0 / 0.05)**2 
    
    # Tighten sig_decay
    if (0.1 < sig_decay < 3.0 and 0.01 < sig_int < 0.2):
        return lp
    
    return -np.inf

def lnprob(theta, X_raw, y, yerr, ddlr):
    lp = lnprior(theta)
    if not np.isfinite(lp): return -np.inf
    return lp + lnlike(theta, X_raw, y, yerr, ddlr)

# MCMC running
data_args = (X_center, y, yerr, ddlr)
nwalkers, niter = 64, 20000
initial = np.array([-0.05, 0.00, 0.02, 0.0, 1.0, 0.1])
ndim = len(initial)
p0 = [initial + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data_args)

print("Running MCMC with DDLR fading...")
sampler.run_mcmc(p0, niter, progress=True)

# RESULTS
samples = sampler.get_chain(discard=5000, thin=20, flat=True)

labels_latex = [r'$\gamma_{\text{Mass}}$', r'$\gamma_{\text{Age}}$', 
                r'$\gamma_{\text{Metal}}$', r'$T_0$', 
                r'$\sigma_{\text{decay}}$', r'$\sigma_{\text{int}}$']

fig = corner.corner(
    samples, labels=labels_latex, quantiles=[0.16, 0.5, 0.84],
    show_titles=True, title_fmt=".3f", color='black', smooth=1.0
)


# Summary
labels_plain = ["gamma_mass", "gamma_age", "gamma_metal", "T0", "sig_decay", "sig_int"]
print("\nPhysical Interpretation:")
for i in range(ndim):
    q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
    err = (q84 - q16) / 2
    sig = abs(q50) / err
    print(f"{labels_plain[i]:>12}: {q50:+.3f} ± {err:.3f} mag ({sig:.2f}σ)")

out_dir = ROOT_DIR / "outputs"
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "MCMC_Corner_Plot_DDLR.png", dpi=150)
plt.show()


# - - - - - - - - - - - - - - - - - - - - - - stuff to work on

print("\nEstimating autocorrelation time...")

try:
    tau = sampler.get_autocorr_time()
    print("Autocorrelation times:")
    for name, t in zip(labels_plain, tau):
        print(f"{name:>12}: {t:.1f} steps")

    print(f"\nRecommended burn-in: {int(3*np.max(tau))}")
    print(f"Effective samples per walker: {niter / tau.mean():.1f}")

except Exception as e:
    print("Autocorrelation estimation failed (chain may be too short).")


theta_med = np.median(samples, axis=0)

# predicted model
model_pred = model(theta_med[:5], X_center, ddlr)

# isolate host contribution
host_effect = model_pred - theta_med[3]   # subtract T0

# measure distance from galaxy center
ddlr_abs = np.abs(ddlr)

from scipy.stats import spearmanr

corr, pval = spearmanr(ddlr_abs, np.abs(host_effect))

print("\nDDLR Fading Diagnostic")
print("----------------------")
print(f"Median sigma_decay: {theta_med[4]:.3f}")
print(f"Spearman correlation between |DDLR| and |host correction|: {corr:.3f}")
print(f"p-value: {pval:.4f}")



# COSMOLOGY TEST (not complete)

print("\nCosmology Impact Test")
print("---------------------")

# posterior median parameters
theta_med = np.median(samples, axis=0)

# predicted host correction
model_pred = model(theta_med[:5], X_center, ddlr)

# isolate host contribution
host_corr = model_pred - theta_med[3]

# apply correction to distance modulus
mu_original = df["MU"].values
mu_corrected = mu_original - host_corr

delta_mu = mu_corrected - mu_original

print(f"Mean distance shift: {np.mean(delta_mu):.4f} mag")
print(f"RMS host correction: {np.sqrt(np.mean(delta_mu**2)):.4f} mag")
print(f"Std deviation of correction: {np.std(delta_mu):.4f} mag")
print(f"Max absolute shift: {np.max(np.abs(delta_mu)):.4f} mag")


# REDSHIFT TEST

from scipy.stats import spearmanr

corr_z, p_z = spearmanr(df["zHD"], delta_mu)

print("\nRedshift Dependence Test")
print("------------------------")
print(f"Spearman correlation (z vs correction): {corr_z:.3f}")
print(f"p-value: {p_z:.4f}")


# SCATTER TEST 

residual_before = df["hubble_residual"].values
residual_after = residual_before - host_corr

scatter_before = np.std(residual_before)
scatter_after = np.std(residual_after)

print("\nDistance Scatter Test")
print("---------------------")
print(f"Scatter before correction: {scatter_before:.4f} mag")
print(f"Scatter after correction : {scatter_after:.4f} mag")
print(f"Scatter reduction        : {scatter_before - scatter_after:.4f} mag")



plt.figure(figsize=(7,5))

plt.scatter(df["zHD"], delta_mu, s=10, alpha=0.5)

plt.axhline(0)

plt.xlabel("Redshift")
plt.ylabel("Host correction (mag)")
plt.title("Cosmological Distance Shift from Host Model")

plt.show()