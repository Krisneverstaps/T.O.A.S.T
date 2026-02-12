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
df_hd = df_hd[df_hd["PROBIA_BEAMS"] > 0.95]

# MERGE EUCLID WITH DES (DES_ID_x = CID)
df = df_euclid[["DES_ID_x", "SFR"]].merge(
    df_hd[["CID_num", "CID", "zHD", "MU", "MUERR", "PROBIA_BEAMS"]],
    left_on="DES_ID_x", right_on="CID_num", how="inner"
)
df = df.merge(df_meta[["CID", "mB", "x1", "c", "x0", "biasCor_mu"]], on="CID", how="left")
df = df.dropna(subset=["zHD", "mB", "x1", "c", "SFR", "MUERR", "x0"])

# PHYSICS
df = calculate_physics(df)

# ------------------------------------------------------------
# SCAN LOWER SFR CUT TO FIND SIGNIFICANCE THRESHOLD
# ------------------------------------------------------------
alpha = 0.05
min_n_per_bin = 5

sfr_cuts = np.linspace(df["SFR"].min(), df["SFR"].quantile(0.2), 100)
significant_cut = None
significant_p = None
significant_t = None

for cut in sfr_cuts:
    df_cut = df[df["SFR"] > cut]

    if len(df_cut) < 2 * min_n_per_bin:
        continue

    sfr_split = df_cut["SFR"].median()
    low_df = df_cut[df_cut["SFR"] < sfr_split]
    high_df = df_cut[df_cut["SFR"] >= sfr_split]

    if len(low_df) < min_n_per_bin or len(high_df) < min_n_per_bin:
        continue

    stats_tmp = run_stats(
        low_df["hubble_residual"],
        high_df["hubble_residual"]
    )

    if stats_tmp["t_p"] < alpha:
        significant_cut = cut
        significant_p = stats_tmp["t_p"]
        significant_t = stats_tmp["t_stat"]
        break

print()
print("=" * 60)
print("SFR lower-cut significance scan")
print("=" * 60)

if significant_cut is None:
    print("Result: never significant")
else:
    print(f"First significant at SFR > {significant_cut:.2f}")
    print(f"t-statistic: {significant_t:.4f}")
    print(f"p-value:     {significant_p:.6f}")
print("=" * 60)
