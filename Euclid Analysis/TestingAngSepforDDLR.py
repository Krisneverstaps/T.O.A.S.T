import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from data_handler import load_snana_format

# SETUP
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
euclid_path = ROOT_DIR / "data" / "Euclid+data.csv"
des_path = ROOT_DIR / "data" / "DES-Dovekie_Metadata.csv"

df_euclid = pd.read_csv(euclid_path)
df_des = load_snana_format(des_path)



df_euclid["DES_ID_x"] = pd.to_numeric(df_euclid["DES_ID_x"], errors="coerce")
df_des["CID"] = pd.to_numeric(df_des["CID"], errors="coerce")


df_merged = df_euclid.merge(
    df_des[["CID", "HOST_DDLR"]],
    left_on="DES_ID_x",
    right_on="CID",
    how="inner"
)


df_merged = df_merged.dropna(subset=["DLR", "HOST_DDLR"])


print(df_merged)

plt.figure(figsize=(8, 8))


max_val = max(df_merged["DLR"].max(), df_merged["HOST_DDLR"].max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='1:1 Agreement')

# Plot the data
plt.scatter(
    df_merged["HOST_DDLR"], 
    df_merged["DLR"], 
    alpha=0.5, 
    color='blue', 
    edgecolor='k',
    label='SNe Samples'
)

# Labeling
plt.title("Comparison of Angular Separation: DES vs. Euclid")
plt.xlabel("DES HOST_DDLR (arcsec)")
plt.ylabel("Euclid Calculated DLR (arcsec)")
plt.legend()
plt.grid(True, alpha=0.3)

# STATISTICS
correlation = df_merged["HOST_DDLR"].corr(df_merged["DLR"])
print(f"Merged {len(df_merged)} supernovae.")
print(f"Correlation coefficient: {correlation:.4f}")

plt.tight_layout()

out_dir = ROOT_DIR / "outputs"
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "DLRComparison.png", dpi=150)
plt.show()
