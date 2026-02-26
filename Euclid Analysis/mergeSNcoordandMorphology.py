import pandas as pd
from pathlib import Path

# PATHS
ROOT_DIR = Path(__file__).resolve().parent.parent
morph_path = (ROOT_DIR / "data" / "with_dlr.csv").resolve()
des_coords_path = (ROOT_DIR / "data" / "DESSNe.csv").resolve()

# LOAD MORPHOLOGY DATA
# load the host galaxy data and the ID key
df_morph = pd.read_csv(morph_path, thousands=',')
df_morph["DES_ID_x"] = pd.to_numeric(df_morph["DES_ID_x"].astype(str).str.replace('"', '').str.strip(), errors="coerce")

# LOAD SN COORDINATE DATA
df_des_coords = pd.read_csv(des_coords_path)

# ID is numeric 
df_des_coords["SNID"] = pd.to_numeric(df_des_coords["SNID"], errors="coerce")

# Rename RA/DEC as the Supernova positions 
df_des_coords = df_des_coords.rename(columns={
    "RA": "SN_RA",
    "DEC": "SN_DEC"
})


# This attaches the SN RA and DEC to the morphology rows based on the ID
df_merged = df_morph.merge(
    df_des_coords[["SNID", "SN_RA", "SN_DEC"]],
    left_on="DES_ID_x",
    right_on="SNID",
    how="inner" # Inner ensures we only keep rows where we have both morphology and coordinates
)

# Remove the SNID column
df_merged = df_merged.drop(columns=["SNID"])

# 5. PRINT SUMMARY AND SAVE
print(f"Successfully merged {len(df_merged)} rows.")
print(df_merged[["DES_ID_x", "SN_RA", "SN_DEC"]].head())

df_merged.to_csv("Morphology_With_SN_Coords.csv", index=False)