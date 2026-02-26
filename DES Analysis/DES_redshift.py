import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from astropy.cosmology import FlatLambdaCDM
from data_handler import download_file, load_snana_format
from analysis_tools import OMEGA_M, binned_weighted_mean2

# LOAD DATA (HD file only — use MU from HD so residuals are centered at zero, like lab 23.1)
hd_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_HD.csv")
df = load_snana_format(hd_path)
df = df[df["PROBIA_BEAMS"] > 0.95].dropna(subset=["zHD", "MU", "MUERR"])

# HUBBLE RESIDUAL = MU - mu_expected (same as lab 23.1; points center at zero)
cosmo = FlatLambdaCDM(H0=70, Om0=OMEGA_M)
df["mu_expected"] = cosmo.distmod(df["zHD"]).value
df["hubble_residual"] = df["MU"] - df["mu_expected"]
scatter = np.std(df["hubble_residual"])

# BINNED WEIGHTED MEAN (no per-point error bars)
bin_centers, bin_means, bin_errs = binned_weighted_mean2(
    df["zHD"].values, df["hubble_residual"].values, df["MUERR"].values, bins=6
)
valid = ~np.isnan(bin_means)

# PRINT SUMMARY
print("=" * 60)
print("Hubble Residual vs Redshift")
print("=" * 60)
print(f"Redshift range: {df['zHD'].min():.4f} -- {df['zHD'].max():.4f}")
print(f"Total SNe: {len(df)}")
print(f"Scatter (std): {scatter:.4f} mag")
print()
print("Binned weighted means:")
print(f"  {'Bin center (z)':<18} {'Weighted mean (mag)':<22} {'Uncertainty (mag)':<18}")
print("-" * 60)
for i in np.where(valid)[0]:
    print(f"  {bin_centers[i]:<18.4f} {bin_means[i]:<22.4f} {bin_errs[i]:<18.4f}")
print("=" * 60)

# PLOT
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["zHD"], df["hubble_residual"], alpha=0.3, color="blue", s=15, label="SNe Ia")
ax.axhline(0, color="black", linestyle="--", label="Zero residual")
ax.errorbar(
    bin_centers[valid], bin_means[valid], yerr=bin_errs[valid], fmt="o", color="red",
    capsize=4, capthick=2, markersize=8, label="Binned weighted mean",
)
ax.set_title(f"Hubble Residual vs Redshift (Scatter: {scatter:.3f} mag)")
ax.set_xlabel("Redshift")
ax.set_ylabel("Hubble Residual (mag)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_dir = Path(__file__).resolve().parent.parent / "outputs"
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "DES_HRvsRedshift.png", dpi=150)
plt.show()
