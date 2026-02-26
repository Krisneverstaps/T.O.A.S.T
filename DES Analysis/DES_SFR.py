import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from data_handler import download_file, load_snana_format
from analysis_tools import calculate_physics2, get_weighted_stats2, run_stats, binned_weighted_mean2

# LOAD DATA
hd_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_HD.csv")
meta_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_Metadata.csv")
df = load_snana_format(hd_path).merge(load_snana_format(meta_path)[['CID', 'HOST_LOGSFR', 'mB', 'x1', 'c', "x0", "biasCor_mu"]], on='CID')
df = df[df['PROBIA_BEAMS'] > 0.95].dropna(subset=['zHD', 'mB', 'x1', 'c', 'HOST_LOGSFR', 'MUERR', "x0"])

# PHYSICS
df = calculate_physics2(df)

df = df[df["HOST_LOGSFR"] > -5]

# SFR STEP 
sfr_split = df['HOST_LOGSFR'].median()
low_df = df[df['HOST_LOGSFR'] < sfr_split]
high_df = df[df['HOST_LOGSFR'] >= sfr_split]

w_mean_low, w_err_low = get_weighted_stats2(low_df['hubble_residual'], low_df['MUERR'])
w_mean_high, w_err_high = get_weighted_stats2(high_df['hubble_residual'], high_df['MUERR'])
sfr_step = w_mean_high - w_mean_low
stats = run_stats(low_df['hubble_residual'], high_df['hubble_residual'])

# BINNED WEIGHTED MEAN
bin_centers, bin_means, bin_errs = binned_weighted_mean2(
    df['HOST_LOGSFR'].values, df['hubble_residual'].values, df['MUERR'].values, bins=6
)
valid = ~np.isnan(bin_means)

# PRINT SUMMARY 
print("=" * 60)
print("Hubble Residual vs Host SFR (SFR Step)")
print("=" * 60)
print(f"HOST_LOGSFR range: {df['HOST_LOGSFR'].min():.2f} -- {df['HOST_LOGSFR'].max():.2f}")
print(f"Total SNe: {len(df)}")
print()
print(f"Low-SFR hosts (< {sfr_split:.2f}):   {len(low_df)} SNe")
print(f"High-SFR hosts (>= {sfr_split:.2f}):  {len(high_df)} SNe")
print()
print(f"Mean residual (low-SFR):   {w_mean_low:+.4f} ± {w_err_low:.4f} mag")
print(f"Mean residual (high-SFR):  {w_mean_high:+.4f} ± {w_err_high:.4f} mag")
print(f"SFR step (high - low):     {sfr_step:+.4f} mag")
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
print(f"  {'Bin center (log SFR)':<22} {'Weighted mean (mag)':<22} {'Uncertainty (mag)':<18}")
print("-" * 65)
for i in np.where(valid)[0]:
    print(f"  {bin_centers[i]:<22.2f} {bin_means[i]:<22.4f} {bin_errs[i]:<18.4f}")
print("=" * 60)

# PLOT 
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['HOST_LOGSFR'], df['hubble_residual'], alpha=0.2, color='gray', s=15, label='SNe Ia')
ax.errorbar(bin_centers[valid], bin_means[valid], yerr=bin_errs[valid], fmt='o', color='blue',
            capsize=4, capthick=2, markersize=8, label='Binned weighted mean')

# horizontal lines
x_min, x_max = df['HOST_LOGSFR'].min(), df['HOST_LOGSFR'].max()
ax.hlines(w_mean_low, x_min, sfr_split, colors='red', lw=2, label=f'Low SFR: {w_mean_low:.3f} ± {w_err_low:.3f}')
ax.hlines(w_mean_high, sfr_split, x_max, colors='orange', lw=2, label=f'High SFR: {w_mean_high:.3f} ± {w_err_high:.3f}')

ax.fill_between([x_min, sfr_split], w_mean_low - w_err_low, w_mean_low + w_err_low, color='red', alpha=0.2)
ax.fill_between([sfr_split, x_max], w_mean_high - w_err_high, w_mean_high + w_err_high, color='orange', alpha=0.2)

ax.axvline(sfr_split, color='black', linestyle='--')
ax.set_title(f'Hubble Residual vs SFR (SFR step: {sfr_step:.3f} mag)')
ax.set_xlabel(r'$\log_{10}(\mathrm{SFR} / M_\odot \mathrm{yr}^{-1})$')
ax.set_ylabel('Hubble Residual (mag)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_dir = Path(__file__).parent.parent / "outputs"
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "DES_HRvsSFR.png", dpi=150)
plt.show()