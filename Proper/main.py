import matplotlib.pyplot as plt
import numpy as np
from data_handler import download_file, load_snana_format
from analysis_tools import calculate_physics, get_weighted_stats, run_stats

# LOAD DATA
hd_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_HD.csv")
meta_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_Metadata.csv")
df = load_snana_format(hd_path).merge(load_snana_format(meta_path)[['CID', 'HOST_LOGMASS', 'mB', 'x1', 'c']], on='CID')
df = df[df['PROBIA_BEAMS'] > 0.95].dropna(subset=['zHD', 'mB', 'x1', 'c', 'HOST_LOGMASS', 'MUERR'])

# PHYSICS & SCATTER COMPARISON
df = calculate_physics(df)
scatter_no_mass = np.std(df['hubble_residual'])

if 'MU' in df.columns:
    df['res_with_mass'] = df['MU'] - df['mu_expected']
    scatter_with_mass = np.std(df.dropna(subset=['res_with_mass'])['res_with_mass'])
    reduction = (scatter_no_mass - scatter_with_mass) / scatter_no_mass * 100
    print(f"Scatter Reduction: {reduction:.2f}%")

# MASS STEP ANALYSIS (WEIGHTED)
low_df = df[df['HOST_LOGMASS'] < 10]
high_df = df[df['HOST_LOGMASS'] >= 10]

w_mean_low, w_err_low = get_weighted_stats(low_df['hubble_residual'], low_df['MUERR'])
w_mean_high, w_err_high = get_weighted_stats(high_df['hubble_residual'], high_df['MUERR'])
p_vals = run_stats(low_df['hubble_residual'], high_df['hubble_residual'])

# PLOTTING
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot Residual vs Redshift
ax1.errorbar(df['zHD'], df['hubble_residual'], yerr=df['MUERR'], fmt='o', alpha=0.3, color='blue', zorder = 1)
ax1.axhline(0, color='black', linestyle='--', zorder = 2, label = 'Zero residual')
ax1.set_title(f'Hubble Residual vs Redshift (Scatter: {scatter_no_mass:.3f} mag)')
ax1.set_xlabel('Redshift ')
ax1.set_ylabel('Hubble Residual (mag)')
ax1.legend()

# Plot Mass Step
ax2.errorbar(df['HOST_LOGMASS'], df['hubble_residual'], yerr=df['MUERR'], fmt='o', alpha=0.2, color='gray', zorder = 1 )
ax2.hlines(w_mean_low, 8, 10, colors='red', lw=2, label=f'Low Mass Mean: {w_mean_low:.3f}', zorder = 3)
ax2.hlines(w_mean_high, 10, 12, colors='orange', lw=2, label=f'High Mass Mean: {w_mean_high:.3f}', zorder = 3)
ax2.fill_between([8, 10], w_mean_low - w_err_low, w_mean_low + w_err_low, color='red', alpha=0.2, zorder = 2)
ax2.fill_between([10, 12], w_mean_high - w_err_high, w_mean_high + w_err_high, color='orange', alpha=0.2, zorder = 2)
ax2.axvline(10, color='black', linestyle='--')
ax2.set_title(f'Mass Step: {w_mean_high - w_mean_low:.3f} mag (t-test p: {p_vals["t_p"]:.4f})')
ax2.set_xlabel('$\log_{10}(M_{host}/M_{\odot}) $')
ax2.set_ylabel('Hubble Residual (mag)')
ax2.legend()

plt.tight_layout()
plt.show()
