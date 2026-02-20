import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import emcee
import corner
from pathlib import Path
from data_handler import download_file, load_snana_format
from analysis_tools import calculate_physics
import seaborn as sns

# PATHS
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
euclid_path = (ROOT_DIR / "data" / "Q1 euclid data.csv").resolve()

euclid_path = Path(__file__).resolve().parent.parent / "data" / "Q1 euclid data.csv"
hd_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_HD.csv")
meta_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_Metadata.csv")

# LOAD DES
df_hd = load_snana_format(hd_path).query("PROBIA_BEAMS > 0.999999")
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

# MERGE
df = (df_euclid.merge(df_hd, on="CID_num")
               .merge(df_meta, on="CID")
               .dropna(subset=["zHD", "mB", "x1", "c", "AGE", "METALLICITY", "LOGMASS", "MUERR"]))

df = calculate_physics(df)

# --- 1. DATA PREPARATION & STANDARDIZATION ---
# Assume df is loaded and cleaned as per your previous script
features = ['LOGMASS', 'AGE', 'METALLICITY']
X_raw = df[features].values
b_des = df['biasCor_mu'].values
b_err = df['biasCorErr_mu'].values # Ensure this matches your column name

# Check for Multi-collinearity (Correlation Matrix)
corr_matrix = df[features].corr()
print("Correlation Matrix of Host Properties:")
print(corr_matrix)

# STANDARDIZE: (X - mean) / std
# This makes the MCMC much more stable
X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
X_scaled = (X_raw - X_mean) / X_std

# --- 2. THE IMPROVED MODEL ---
def bias_transfer_model(theta, X):
    """
    theta: [beta_mass, alpha_age, gamma_met, offset]
    All parameters are now 'standardized' coefficients.
    """
    beta, alpha, gamma, offset = theta
    # X[:, 0]=Mass, X[:, 1]=Age, X[:, 2]=Metallicity
    delta_bias = (beta * X[:, 0] + 
                  alpha * X[:, 1] + 
                  gamma * X[:, 2] + 
                  offset)
    return delta_bias

def lnlike_bias(theta, X, b_des, b_err):
    delta_pred = bias_transfer_model(theta, X)
    # Using a Gaussian likelihood
    return -0.5 * np.sum(((b_des - delta_pred) / b_err)**2)

def lnprior_bias(theta):
    # Since data is scaled, coefficients are usually small (between -0.5 and 0.5)
    if all(-1.0 < t < 1.0 for t in theta):
        return 0.0
    return -np.inf

def lnprob_bias(theta, X, b_des, b_err):
    lp = lnprior_bias(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_bias(theta, X, b_des, b_err)

# --- 3. MCMC EXECUTION ---
ndim = 4
nwalkers = 64
niter = 10000

# Start walkers in a small ball around zero
initial_guess = np.zeros(ndim)
p0 = [initial_guess + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_bias, args=(X_scaled, b_des, b_err))

print("Running MCMC...")
# Burn-in
pos, prob, state = sampler.run_mcmc(p0, 5000, progress=True)
sampler.reset()
# Production
sampler.run_mcmc(pos, niter, progress=True)

# Get samples
samples = sampler.get_chain(flat=True)

# --- 4. SUMMARY & PHYSICAL INTERPRETATION ---
# To get the "Physical" coefficients back, we divide the standardized 
# coefficients by the original standard deviation.
q50 = np.median(samples, axis=0)
physical_coeffs = q50[:3] / X_std

summary_data = {
    "Parameter": ["Mass Scale (Beta)", "Age Scale (Alpha)", "Metal Scale (Gamma)", "Offset"],
    "Standardized Coeff": q50,
    "Physical Coeff (per unit)": list(physical_coeffs) + [q50[3]]
}
print("\n--- MCMC Results ---")
print(pd.DataFrame(summary_data))

# --- 5. PREDICT EUCLID BIAS ---
# Use the best-fit standardized model
best_theta = q50
df['bias_improvement'] = bias_transfer_model(best_theta, X_scaled)
df['biasCor_mu_EUCLID'] = df['biasCor_mu'] - df['bias_improvement']

# --- 6. VISUALIZATION ---
# Corner Plot
labels = [r"$\beta_{Mass}$", r"$\alpha_{Age}$", r"$\gamma_{Metal}$", "Offset"]
fig = corner.corner(samples, labels=labels, truths=best_theta, 
                    show_titles=True, title_fmt=".4f")
plt.show()

# Correlation Heatmap (for the report)
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Property Correlations (Check for Collinearity)")
plt.show()


import pandas as pd
import numpy as np

# Load the Euclid dataset (Q1 Euclid data CSV)
df_euclid = pd.read_csv(euclid_path)

# Assuming the MCMC best-fit parameters (standardized coefficients)
best_theta = [-0.01768849, 0.00763273, 0.00437609, 0.01602126]  # Example values from MCMC

# Assuming the uncertainty in the model parameters (these should come from MCMC)
sigma_beta = 0.0004  # Example uncertainty for beta
sigma_alpha = 0.0004  # Example uncertainty for alpha
sigma_gamma = 0.0003  # Example uncertainty for gamma
sigma_offset = 0.0003  # Example uncertainty for offset

# Extract standardized features from the dataset (LOGMASS, AGE, METALLICITY)
# Ensure these columns are present in your Euclid dataset and standardized appropriately
X_scaled = df_euclid[['HOST_LOGMASS', 'phz_pp_mode_sfhage', 'phz_pp_mode_stellarmetallicity']].values

# Define the bias transfer model
def bias_transfer_model(theta, X):
    beta, alpha, gamma, offset = theta
    delta_bias = (beta * X[:, 0] + alpha * X[:, 1] + gamma * X[:, 2] + offset)
    return delta_bias

# Calculate the bias improvement for each object
bias_improvement = bias_transfer_model(best_theta, X_scaled)

# Calculate uncertainty in the bias improvement using error propagation
def bias_uncertainty(X, sigma_beta, sigma_alpha, sigma_gamma, sigma_offset):
    # Propagate uncertainty based on the features and parameter uncertainties
    uncertainty = np.sqrt(
        (X[:, 0] * sigma_beta)**2 + 
        (X[:, 1] * sigma_alpha)**2 + 
        (X[:, 2] * sigma_gamma)**2 + 
        (sigma_offset)**2
    )
    return uncertainty

# Calculate the uncertainty in the bias improvement for each object
bias_improvement_uncertainty = bias_uncertainty(X_scaled, sigma_beta, sigma_alpha, sigma_gamma, sigma_offset)

# Now, subtract the bias improvement from the observed bias (biasCor_mu) to get the corrected bias
df_euclid['bias_improvement'] = bias_improvement
df_euclid['biasCor_mu_EUCLID'] = df['biasCor_mu'] - df['bias_improvement']

# Assuming you have the uncertainty in the observed bias (e.g., 'biasCorErr_mu'), add uncertainty in the corrected bias
df_euclid['biasCor_mu_EUCLID_uncertainty'] = df['biasCorErr_mu'] + bias_improvement_uncertainty

# Print out the relevant columns
print("Printing out the bias and uncertainty columns:")
print(df_euclid[['CID', 'biasCor_mu', 'bias_improvement', 'biasCor_mu_EUCLID', 'biasCor_mu_EUCLID_uncertainty']])

# Save the updated dataset with the new bias and uncertainty columns to the same CSV file
df_euclid.to_csv(euclid_path, index=False)

print(f"Bias correction and uncertainty results saved to {euclid_path}")