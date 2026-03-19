import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from data_handler import download_file, load_snana_format
from analysis_tools import calculate_physics2, get_weighted_stats2, run_stats, binned_weighted_mean2

# EUCLID CSV PATH 
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
euclid_path = (ROOT_DIR / "data" / "Q1 euclid data.csv").resolve()
euclid_morph_path = (ROOT_DIR / "data" / "Euclid+data.csv").resolve()

# LOAD EUCLID DATA
df_euclid = pd.read_csv(euclid_path)
df_euclid["DES_ID_x"] = pd.to_numeric(df_euclid["DES_ID_x"], errors="coerce")

m_col = "logmass" if "logmass" in df_euclid.columns else "HOST_LOGMASS"
df_euclid["HOST_LOGMASS"] = pd.to_numeric(df_euclid[m_col], errors="coerce")


#Load morphology and remove the commas from numbers like 1,000,000
df_morph = pd.read_csv(euclid_morph_path, thousands=',')


# convert to strings and remove non-numeric characters (excecpt ID)
df_morph["DES_ID_x"] = df_morph["DES_ID_x"].astype(str).str.replace('"', '').str.strip()
df_morph["DES_ID_x"] = pd.to_numeric(df_morph["DES_ID_x"], errors="coerce")

# Convert DDLR to numeric 
df_morph["DDLR"] = pd.to_numeric(df_morph["DDLR"], errors="coerce")

# ADD DDLR TO EUCLID DATA 
df_euclid = df_euclid[["DES_ID_x", "HOST_LOGMASS"]].merge(
    df_morph[["DES_ID_x", "DDLR"]], 
    on="DES_ID_x", 
    how="inner"
)

# LOAD DES DATA
hd_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_HD.csv")
meta_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_Metadata.csv")
df_hd = load_snana_format(hd_path)
df_meta = load_snana_format(meta_path)

# Prepare DES HD
df_hd["CID_num"] = pd.to_numeric(df_hd["CID"], errors="coerce")
df_hd = df_hd.dropna(subset=["CID_num", "zHD", "MU", "MUERR"]).copy()
df_hd = df_hd[df_hd["PROBIA_BEAMS"] > 0.95]


# Merge Euclid with DES HD
df = df_euclid[["DES_ID_x", "HOST_LOGMASS", "DDLR"]].merge(
    df_hd[["CID_num", "CID", "zHD", "MU", "MUERR", "PROBIA_BEAMS"]],
    left_on="DES_ID_x", right_on="CID_num", how="inner"
)

# Merge with Metadata
df = df.merge(df_meta[["CID", "mB", "x1", "c", "x0"]], on="CID", how="left")
df = df.dropna(subset=["zHD", "mB", "x1", "c", "HOST_LOGMASS", "MUERR", "x0", "DDLR"])

# PHYSICS
df = calculate_physics2(df)

# FILTER
df = df[df["DDLR"] < 4]

def compute_ddlr_step(group_df, title_suffix, threshold=1):
    """Compute and plot Hubble residual vs DDLR for a given group."""
    # SPLIT DATA BASED ON THRESHOLD
    low_df = group_df[group_df['DDLR'] < threshold]
    high_df = group_df[group_df['DDLR'] >= threshold]

    # WEIGHTED STATS
    w_mean_low, w_err_low = get_weighted_stats2(low_df['hubble_residual'], low_df['MUERR'])
    w_mean_high, w_err_high = get_weighted_stats2(high_df['hubble_residual'], high_df['MUERR'])
    ddlr_step = w_mean_high - w_mean_low
    stats = run_stats(low_df['hubble_residual'], high_df['hubble_residual'])

    # BINNED WEIGHTED MEAN
    bin_centers, bin_means, bin_errs = binned_weighted_mean2(
        group_df['DDLR'].values, group_df['hubble_residual'].values, group_df['MUERR'].values, bins=6
    )
    valid = ~np.isnan(bin_means)

    print("=" * 60)
    print(f"Hubble Residual vs DDLR ({title_suffix})")
    print("=" * 60)

    print(f"DDLR range: "
          f"{group_df['DDLR'].min():.2f} -- "
          f"{group_df['DDLR'].max():.2f}")
    print(f"Total SNe: {len(group_df)}")
    print()

    print(f"Close to host (DDLR < {threshold}):  {len(low_df)} SNe")
    print(f"Far from host (DDLR >= {threshold}): {len(high_df)} SNe")
    print()

    print(f"Mean residual (Close):  {w_mean_low:+.4f} ± {w_err_low:.4f} mag")
    print(f"Mean residual (Far):    {w_mean_high:+.4f} ± {w_err_high:.4f} mag")
    print(f"DDLR step (Far − Close): {ddlr_step:+.4f} mag")
    print()

    print("Two-sample t-test:")
    print(f"  t-statistic: {stats['t_stat']:.4f}")
    print(f"  p-value:     {stats['t_p']:.6f}")
    if stats['t_p'] < 0.05:
        print("  Statistically significant (p < 0.05)")
    else:
        print("  Not statistically significant (p ≥ 0.05)")
    print()

    print("Kolmogorov–Smirnov test:")
    print(f"  KS statistic: {stats['ks_stat']:.4f}")
    print(f"  p-value:      {stats['ks_p']:.6f}")
    if stats['ks_p'] < 0.05:
        print("  Distributions differ (p < 0.05)")
    else:
        print("  Distributions do not differ significantly (p ≥ 0.05)")
    print()

    print("Binned weighted means:")
    print(f"  {'Bin center (DDLR)':<22}"
          f"{'Weighted mean (mag)':<22}"
          f"{'Uncertainty (mag)':<18}")
    print("-" * 65)
    for i in np.where(valid)[0]:
        print(f"  {bin_centers[i]:<22.2f}"
              f"{bin_means[i]:<22.4f}"
              f"{bin_errs[i]:<18.4f}")

    print("=" * 60)

    # PLOT (Matches your previous style)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(group_df['DDLR'], group_df['hubble_residual'], alpha=0.2, color='gray', s=15, label='SNe Ia')
    ax.errorbar(bin_centers[valid], bin_means[valid], yerr=bin_errs[valid], fmt='o', color='blue',
                capsize=4, capthick=2, markersize=8, label='Binned weighted mean')
    
    # Plot Step Lines
    ax.hlines(w_mean_low, group_df['DDLR'].min(), threshold, colors='red', lw=2, label=f'Close: {w_mean_low:.3f} ± {w_err_low:.3f}')
    ax.hlines(w_mean_high, threshold, group_df['DDLR'].max(), colors='orange', lw=2, label=f'Far: {w_mean_high:.3f} ± {w_err_high:.3f}')
    
    # Uncertainty shading
    ax.fill_between([group_df['DDLR'].min(), threshold], w_mean_low - w_err_low, w_mean_low + w_err_low, color='red', alpha=0.2)
    ax.fill_between([threshold, group_df['DDLR'].max()], w_mean_high - w_err_high, w_mean_high + w_err_high, color='orange', alpha=0.2)
    
    ax.axvline(threshold, color='black', linestyle='--')
    ax.set_title(f'Hubble Residual vs DDLR ({title_suffix}, Step: {ddlr_step:.3f} mag)')
    ax.set_xlabel('DDLR')
    ax.set_ylabel('Hubble Residual (mag)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / f"Euc_DDLR_step_{title_suffix.replace(' ', '_')}.png", dpi=150)
    plt.show()

compute_ddlr_step(df, "Full Sample")