import numpy as np
from scipy.stats import ttest_ind, ks_2samp
from astropy.cosmology import FlatLambdaCDM

# DES 5-year cosmology and standardization (4_DISTANCES_COVMAT README)
OMEGA_M = 0.330       # ± 0.015
ALPHA = 0.169         # ± 0.0003
BETA = 3.14           # ± 0.04
GAMMA = 0.033         # ± 0.008 (mass step; used in HD MU, not in mu_no_mass)
M0_AVG = -29.96210    # absolute magnitude zero point

def calculate_physics(df, alpha=ALPHA, beta=BETA, m0=M0_AVG, omega_m=OMEGA_M):
    """Calculates expectations, residuals, and corrections (DES 5-year parameters)."""
    cosmo = FlatLambdaCDM(H0=70, Om0=omega_m)

    # Expected distance modulus from redshift
    df['mu_expected'] = cosmo.distmod(df['zHD']).value

    # Recalculate MU without mass correction: μ = mB + α*x1 - β*c - bias - M0 (no γ*step)
    df['mu_no_mass'] = -2.5*np.log10(df['x0']) + (alpha * df['x1']) - (beta * df['c']) - (df["biasCor_mu"]) - m0

    # Residual (observed minus expected)
    df['hubble_residual'] = df['mu_no_mass'] - df['mu_expected']
    return df

def get_weighted_stats(residuals, observational_errors, bias_errors, sigma_int=0.11):
    """
    Calculates weighted mean using both observational error and intrinsic scatter.
    """
    # Total variance is the sum of squares of errors
    total_error = np.sqrt(observational_errors**2 + sigma_int**2 + bias_errors*2)
    weights = 1 / (total_error**2)
    
    weighted_mean = np.average(residuals, weights=weights)
    # Uncertainty of the mean
    uncertainty = 1 / np.sqrt(np.sum(weights))
    
    return weighted_mean, uncertainty

def run_stats(low_res, high_res):
    """Runs T-Test and KS-Test between two mass bins."""
    t_stat, t_p = ttest_ind(low_res, high_res, equal_var=False)
    ks_stat, ks_p = ks_2samp(low_res, high_res)
    return {"t_stat": t_stat, "t_p": t_p, "ks_stat": ks_stat, "ks_p": ks_p}

def binned_weighted_mean(x, y, observational_errors, bias_errors,  bins=6, sigma_int=0.11):
    """Returns bin centers, weighted means, and uncertainties per bin."""
    edges = np.linspace(x.min(), x.max(), bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    means = np.full(bins, np.nan)
    errs = np.full(bins, np.nan)
    for i in range(bins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if i == bins - 1:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        if np.sum(mask) > 0:
            means[i], errs[i] = get_weighted_stats(y[mask], observational_errors[mask], bias_errors[mask], sigma_int=sigma_int)
    return centers, means, errs

def get_weighted_stats_with_prob(residuals, observational_errors, probabilities, sigma_int=0.11):
    """
    Calculates weighted mean using:
    1. Inverse Variance (1 / sigma^2)
    2. Classification Probability (PROBIA_BEAMS)
    """
    # Calculate the standard statistical weight (Inverse Variance)
    total_variance = observational_errors**2 + sigma_int**2
    variance_weight = 1 / total_variance
    
    # Combine with the Probability Weight
    # This gives more priority to SNe with high PROBIA_BEAMS
    combined_weights = variance_weight * probabilities
    
    # Weighted Average: 
    weighted_mean = np.sum(residuals * combined_weights) / np.sum(combined_weights)
    
    # Uncertainty of the weighted mean
    uncertainty = 1 / np.sqrt(np.sum(combined_weights))
    
    return weighted_mean, uncertainty

def binned_weighted_mean_with_prob(x, y, errors, probs, bins=6, sigma_int=0.11):
    """Binned version using probability-weight"""
    edges = np.linspace(x.min(), x.max(), bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    means = np.full(bins, np.nan)
    errs = np.full(bins, np.nan)
    
    for i in range(bins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if i == bins - 1:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
            
        if np.sum(mask) > 0:
            means[i], errs[i] = get_weighted_stats_with_prob(
                y[mask], errors[mask], probs[mask], sigma_int=sigma_int
            )
    return centers, means, errs