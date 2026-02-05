import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from data_handler import download_file, load_snana_format
from analysis_tools import calculate_physics, get_weighted_stats, run_stats, binned_weighted_mean

# LOAD DATA
hd_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_HD.csv")
meta_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_Metadata.csv")
df = load_snana_format(hd_path).merge(load_snana_format(meta_path)[['CID', 'HOST_LOGMASS', 'mB', 'x1', 'c', 'x0']], on='CID')
df = df[df['PROBIA_BEAMS'] > 0.95].dropna(subset=['zHD', 'mB', 'x1', 'c', 'x0', 'HOST_LOGMASS', 'MUERR'])

# PHYSICS
df = calculate_physics(df)

# MASS STEP (WEIGHTED)
low_df = df[df['HOST_LOGMASS'] < 10]
high_df = df[df['HOST_LOGMASS'] >= 10]

w_mean_low, w_err_low = get_weighted_stats(low_df['hubble_residual'], low_df['MUERR'])
w_mean_high, w_err_high = get_weighted_stats(high_df['hubble_residual'], high_df['MUERR'])
mass_step = w_mean_high - w_mean_low
stats = run_stats(low_df['hubble_residual'], high_df['hubble_residual'])

# BINNED WEIGHTED MEAN (no per-point error bars)
bin_centers, bin_means, bin_errs = binned_weighted_mean(
    df['HOST_LOGMASS'].values, df['hubble_residual'].values, df['MUERR'].values, bins=6
)
valid = ~np.isnan(bin_means)

# PRINT SUMMARY
print("=" * 60)
print("Hubble Residual vs Host Mass (Mass Step)")
print("=" * 60)
print(f"HOST_LOGMASS range: {df['HOST_LOGMASS'].min():.2f} -- {df['HOST_LOGMASS'].max():.2f}")
print(f"Total SNe: {len(df)}")
print()
print(f"Low-mass hosts (log M* < 10):  {len(low_df)} SNe")
print(f"High-mass hosts (log M* >= 10): {len(high_df)} SNe")
print()
print(f"Mean residual (low-mass):  {w_mean_low:+.4f} ± {w_err_low:.4f} mag")
print(f"Mean residual (high-mass): {w_mean_high:+.4f} ± {w_err_high:.4f} mag")
print(f"Mass step (high - low):    {mass_step:+.4f} mag")
print()
print("Two-sample t-test:")
print(f"  t-statistic: {stats['t_stat']:.4f}")
print(f"  p-value:     {stats['t_p']:.6f}")
if stats['t_p'] < 0.05:
    print("  Statistically significant (p < 0.05)")
else:
    print("  Not statistically significant (p >= 0.05)")
print()
print("Kolmogorov-Smirnov test:")
print(f"  KS statistic: {stats['ks_stat']:.4f}")
print(f"  p-value:      {stats['ks_p']:.6f}")
if stats['ks_p'] < 0.05:
    print("  Distributions differ (p < 0.05)")
else:
    print("  Distributions do not differ significantly (p >= 0.05)")
print()
print("Binned weighted means:")
print(f"  {'Bin center (log M*)':<22} {'Weighted mean (mag)':<22} {'Uncertainty (mag)':<18}")
print("-" * 65)
for i in np.where(valid)[0]:
    print(f"  {bin_centers[i]:<22.2f} {bin_means[i]:<22.4f} {bin_errs[i]:<18.4f}")
print("=" * 60)

# PLOT
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['HOST_LOGMASS'], df['hubble_residual'], alpha=0.2, color='gray', s=15, label='SNe Ia')
ax.errorbar(bin_centers[valid], bin_means[valid], yerr=bin_errs[valid], fmt='o', color='blue',
            capsize=4, capthick=2, markersize=8, label='Binned weighted mean')
ax.hlines(w_mean_low, 8, 10, colors='red', lw=2, label=f'Low mass: {w_mean_low:.3f} ± {w_err_low:.3f}')
ax.hlines(w_mean_high, 10, 12, colors='orange', lw=2, label=f'High mass: {w_mean_high:.3f} ± {w_err_high:.3f}')
ax.fill_between([8, 10], w_mean_low - w_err_low, w_mean_low + w_err_low, color='red', alpha=0.2)
ax.fill_between([10, 12], w_mean_high - w_err_high, w_mean_high + w_err_high, color='orange', alpha=0.2)
ax.axvline(10, color='black', linestyle='--')
ax.set_title(f'Hubble Residual vs Host Mass (Mass step: {mass_step:.3f} mag)')
ax.set_xlabel(r'$\log_{10}(M_{host}/M_\odot)$')
ax.set_ylabel('Hubble Residual (mag)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_dir = Path(__file__).parent.parent / "outputs"
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "hubble_residual_vs_mass_step.png", dpi=150)
plt.show()
