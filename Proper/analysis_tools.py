import numpy as np
from scipy.stats import ttest_ind, ks_2samp
from astropy.cosmology import FlatLambdaCDM

def calculate_physics(df, alpha=0.14, beta=3.1, m0=-19.3):
    """Calculates expectations, residuals, and corrections."""
    cosmo = FlatLambdaCDM(H0=70, Om0=0.33)
    
    # Expected distance modulus from Redshift
    df['mu_expected'] = cosmo.distmod(df['zHD']).value
    
    # Recalculate MU without mass correction: μ = mB + α*x1 - β*c - M0
    df['mu_no_mass'] = df['mB'] + (alpha * df['x1']) - (beta * df['c']) - m0
    
    # Calculate the Residual (Difference between observed and theory)
    df['hubble_residual'] = df['mu_no_mass'] - df['mu_expected']
    return df

def get_weighted_stats(residuals, observational_errors, sigma_int=0.11):
    """
    Calculates weighted mean using both observational error and intrinsic scatter.
    """
    # Total variance is the sum of squares of errors
    total_error = np.sqrt(observational_errors**2 + sigma_int**2)
    weights = 1 / (total_error**2)
    
    weighted_mean = np.average(residuals, weights=weights)
    # Uncertainty of the mean
    uncertainty = 1 / np.sqrt(np.sum(weights))
    
    return weighted_mean, uncertainty

def run_stats(low_res, high_res):
    """Runs T-Test and KS-Test between two mass bins."""
    t_stat, t_p = ttest_ind(low_res, high_res, equal_var=False)
    ks_stat, ks_p = ks_2samp(low_res, high_res)
    return {"t_p": t_p, "ks_p": ks_p}