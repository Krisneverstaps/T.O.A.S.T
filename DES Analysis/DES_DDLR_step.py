import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Changed from 'import pandas' to 'import pandas as pd'
from pathlib import Path
from data_handler import download_file, load_snana_format
from analysis_tools import calculate_physics2, get_weighted_stats2, run_stats, binned_weighted_mean2

# LOAD DATA
hd_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_HD.csv")
meta_path = download_file("4_DISTANCES_COVMAT/DES-Dovekie_Metadata.csv")

# Fixed the list of columns (removed duplicates)
meta_cols = ['CID', 'HOST_DDLR', 'mB', 'x1', 'c', 'x0']

df = load_snana_format(hd_path).merge(
    load_snana_format(meta_path)[meta_cols],
    on='CID'
)

# Clean and filter data
df = df[df['PROBIA_BEAMS'] > 0.95].dropna(subset=['zHD', 'mB', 'x1', 'c', 'HOST_DDLR', 'MUERR', 'x0'])

# PHYSICS
df = calculate_physics2(df)

# FILTER DATA BASED ON HOST_DDLR
df = df[df["HOST_DDLR"] < 4]   # Restricting to SNe within the galaxy region
df = df[df["HOST_DDLR"] > -5]  # Removes the -9 placeholder values


def compute_HOST_DDLR_step(group_df, title_suffix, threshold=1):
    """Compute and plot Hubble residual vs HOST_DDLR with full summary output."""
    
    # SPLIT DATA BASED ON THRESHOLD
    low_df = group_df[group_df['HOST_DDLR'] < threshold]
    high_df = group_df[group_df['HOST_DDLR'] >= threshold]

    # WEIGHTED STATS
    w_mean_low, w_err_low = get_weighted_stats2(low_df['hubble_residual'], low_df['MUERR'])
    w_mean_high, w_err_high = get_weighted_stats2(high_df['hubble_residual'], high_df['MUERR'])
    
    HOST_DDLR_step = w_mean_high - w_mean_low
    stats = run_stats(low_df['hubble_residual'], high_df['hubble_residual'])

    # BINNED WEIGHTED MEAN
    bin_centers, bin_means, bin_errs = binned_weighted_mean2(
        group_df['HOST_DDLR'].values, 
        group_df['hubble_residual'].values, 
        group_df['MUERR'].values, 
        bins=6
    )
    valid = ~np.isnan(bin_means)

    # PRINT SUMMARY (Exact formatting from your original request)
    print("=" * 60)
    print(f"Hubble Residual vs HOST_DDLR ({title_suffix})")
    print("=" * 60)

    print(f"HOST_DDLR range: "
          f"{group_df['HOST_DDLR'].min():.2f} -- "
          f"{group_df['HOST_DDLR'].max():.2f}")
    print(f"Total SNe: {len(group_df)}")
    print()

    print(f"Close to center (HOST_DDLR < {threshold}):  {len(low_df)} SNe")
    print(f"Further out (HOST_DDLR >= {threshold}):    {len(high_df)} SNe")
    print()

    print(f"Mean residual (Low DDLR):   {w_mean_low:+.4f} ± {w_err_low:.4f} mag")
    print(f"Mean residual (High DDLR):  {w_mean_high:+.4f} ± {w_err_high:.4f} mag")
    print(f"HOST_DDLR step (High − Low): {HOST_DDLR_step:+.4f} mag")
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
    print(f"  {'Bin center (HOST_DDLR)':<22}"
          f"{'Weighted mean (mag)':<22}"
          f"{'Uncertainty (mag)':<18}")
    print("-" * 65)
    for i in np.where(valid)[0]:
        print(f"  {bin_centers[i]:<22.2f}"
              f"{bin_means[i]:<22.4f}"
              f"{bin_errs[i]:<18.4f}")

    print("=" * 60)

    # PLOT
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(group_df['HOST_DDLR'], group_df['hubble_residual'], alpha=0.2, color='gray', s=15, label='SNe Ia')
    
    # Binned points
    ax.errorbar(bin_centers[valid], bin_means[valid], yerr=bin_errs[valid], fmt='o', color='blue',
                capsize=4, capthick=2, markersize=8, label='Binned weighted mean')
    
    # Step Lines
    ax.hlines(w_mean_low, group_df['HOST_DDLR'].min(), threshold, colors='red', lw=2, label=f'Close: {w_mean_low:.3f} ± {w_err_low:.3f}')
    ax.hlines(w_mean_high, threshold, group_df['HOST_DDLR'].max(), colors='orange', lw=2, label=f'Far: {w_mean_high:.3f} ± {w_err_high:.3f}')
    
    # Shading
    ax.fill_between([group_df['HOST_DDLR'].min(), threshold], w_mean_low - w_err_low, w_mean_low + w_err_low, color='red', alpha=0.2)
    ax.fill_between([threshold, group_df['HOST_DDLR'].max()], w_mean_high - w_err_high, w_mean_high + w_err_high, color='orange', alpha=0.2)
    
    ax.axvline(threshold, color='black', linestyle='--')
    ax.set_title(f'Hubble Residual vs HOST_DDLR ({title_suffix}, Step: {HOST_DDLR_step:.3f} mag)')
    ax.set_xlabel('HOST_DDLR')
    ax.set_ylabel('Hubble Residual (mag)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / f"Euc_HOST_DDLR_step_{title_suffix.replace(' ', '_')}.png", dpi=150)
    plt.show()

# Execution
compute_HOST_DDLR_step(df, "Full Sample")