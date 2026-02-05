import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from data_handler import download_file, load_snana_format
from analysis_tools import calculate_physics, get_weighted_stats, run_stats, binned_weighted_mean

# EUCLID CSV PATH
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
euclid_path = (ROOT_DIR / "data" / "Q1 euclid data.csv").resolve()

# LOAD EUCLID DATA
df_euclid = pd.read_csv(euclid_path)
df_euclid["DES_ID_x"] = pd.to_numeric(df_euclid["DES_ID_x"], errors="coerce")
df_euclid["zHD"] = pd.to_numeric(df_euclid["zHD"], errors="coerce")
df_euclid["SFR"] = pd.to_numeric(df_euclid["phz_pp_mode_sfr"], errors="coerce")

# LOAD DES DATA
hd_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_HD.csv")
meta_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_Metadata.csv")
df_hd = load_snana_format(hd_path)
df_meta = load_snana_format(meta_path)
df_hd["CID_num"] = pd.to_numeric(df_hd["CID"], errors="coerce")
df_hd = df_hd.dropna(subset=["CID_num", "zHD", "MU", "MUERR"]).copy()
df_hd = df_hd[df_hd["PROBIA_BEAMS"] > 0.999999]

# MERGE EUCLID WITH DES (DES_ID_x = CID)
df = df_euclid[["DES_ID_x", "SFR"]].merge(
    df_hd[["CID_num", "CID", "zHD", "MU", "MUERR", "PROBIA_BEAMS"]],
    left_on="DES_ID_x", right_on="CID_num", how="inner"
)
df = df.merge(df_meta[["CID", "mB", "x1", "c", "x0"]], on="CID", how="left")
df = df.dropna(subset=["zHD", "mB", "x1", "c", "SFR", "MUERR", "x0"])

# PHYSICS
df = calculate_physics(df)

df = df[df["SFR"] > -6.9]

# SFR STEP (split at median SFR)
sfr_split = df["SFR"].median()
low_df = df[df["SFR"] < sfr_split]
high_df = df[df["SFR"] >= sfr_split]
w_mean_low, w_err_low = get_weighted_stats(low_df["hubble_residual"], low_df["MUERR"])
w_mean_high, w_err_high = get_weighted_stats(high_df["hubble_residual"], high_df["MUERR"])
sfr_step = w_mean_high - w_mean_low
stats = run_stats(low_df["hubble_residual"], high_df["hubble_residual"])

# BINNED WEIGHTED MEAN
bin_centers, bin_means, bin_errs = binned_weighted_mean(
    df["SFR"].values, df["hubble_residual"].values, df["MUERR"].values, bins=6
)
valid = ~np.isnan(bin_means)

# PRINT SUMMARY
print("=" * 60)
print("Euclid: Hubble Residual vs SFR (phz_pp_mode_sfr)")
print("=" * 60)
print(f"phz_pp_mode_sfr range: {df['SFR'].min():.2f} -- {df['SFR'].max():.2f}")
print(f"Redshift range: {df['zHD'].min():.4f} -- {df['zHD'].max():.4f}")
print(f"Total SNe: {len(df)}")
print(f"Split at median SFR: {sfr_split:.2f}")
print()
print(f"Low SFR (SFR < {sfr_split:.2f}):  {len(low_df)} SNe")
print(f"High SFR (SFR >= {sfr_split:.2f}): {len(high_df)} SNe")
print()
print(f"Mean residual (low SFR):  {w_mean_low:+.4f} ± {w_err_low:.4f} mag")
print(f"Mean residual (high SFR): {w_mean_high:+.4f} ± {w_err_high:.4f} mag")
print(f"SFR step (high - low):    {sfr_step:+.4f} mag")
print()
print("Two-sample t-test:")
print(f"  t-statistic: {stats['t_stat']:.4f}")
print(f"  p-value:     {stats['t_p']:.6f}")
if stats["t_p"] < 0.05:
    print("  Statistically significant (p < 0.05)")
else:
    print("  Not statistically significant (p >= 0.05)")
print()
print("Kolmogorov-Smirnov test:")
print(f"  KS statistic: {stats['ks_stat']:.4f}")
print(f"  p-value:      {stats['ks_p']:.6f}")
if stats["ks_p"] < 0.05:
    print("  Distributions differ (p < 0.05)")
else:
    print("  Distributions do not differ significantly (p >= 0.05)")
print()
print("Binned weighted means:")
print(f"  {'Bin center (SFR)':<22} {'Weighted mean (mag)':<22} {'Uncertainty (mag)':<18}")
print("-" * 65)
for i in np.where(valid)[0]:
    print(f"  {bin_centers[i]:<22.2f} {bin_means[i]:<22.4f} {bin_errs[i]:<18.4f}")
print("=" * 60)

# PLOT
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["SFR"], df["hubble_residual"], alpha=0.2, color="gray", s=15, label="Euclid SNe Ia")
ax.errorbar(bin_centers[valid], bin_means[valid], yerr=bin_errs[valid], fmt="o", color="blue",
            capsize=2, capthick=1, markersize=3, label="Binned weighted mean")
x_min, x_max = df["SFR"].min(), df["SFR"].max()
ax.hlines(w_mean_low, x_min, sfr_split, colors="red", lw=2, label=f"Low SFR: {w_mean_low:.3f} ± {w_err_low:.3f}")
ax.hlines(w_mean_high, sfr_split, x_max, colors="orange", lw=2, label=f"High SFR: {w_mean_high:.3f} ± {w_err_high:.3f}")
ax.fill_between([x_min, sfr_split], w_mean_low - w_err_low, w_mean_low + w_err_low, color="red", alpha=0.2)
ax.fill_between([sfr_split, x_max], w_mean_high - w_err_high, w_mean_high + w_err_high, color="orange", alpha=0.2)
ax.axvline(sfr_split, color="black", linestyle="--")
ax.set_title(f"Euclid: Hubble Residual vs SFR (phz_pp_mode_sfr, step: {sfr_step:.3f} mag)")
ax.set_xlabel("phz_pp_mode_sfr")
ax.set_ylabel("Hubble Residual (mag)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_dir = ROOT_DIR / "outputs"
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "euclid_hubble_residual_vs_sfr_step.png", dpi=150)
plt.show()
