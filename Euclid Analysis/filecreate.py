import pandas as pd
from pathlib import Path
from data_handler import load_snana_format
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
galaxies = ROOT_DIR / "data" / "Euclid+data.csv"
hosts = ROOT_DIR / "data" / "DES-Dovekie_Metadata.csv"

galaxies = pd.read_csv(galaxies)
hosts = load_snana_format(hosts)



# Clean column names
galaxies.columns = galaxies.columns.str.strip()
hosts.columns = hosts.columns.str.strip()

# Merge using DES_ID_x and CIDint
merged = galaxies.merge(
    hosts[["CIDint", "HOST_DDLR"]],
    left_on="DES_ID_x",
    right_on="CIDint",
    how="left"
)

# Select the final columns
result = merged[
    [
        "Euclid_ID_x",
        "DES_ID_x",
        "DDLR",
        "HOST_DDLR",
        "right_ascension",
        "declination",
        "HOST_RA",
        "HOST_DEC",
        "SN_RA",
        "SN_DEC"
    ]
]

# Rename for clarity
result = result.rename(columns={
    "Euclid_ID_x": "Euclid_ID",
    "DES_ID_x": "DES_ID"
})

# Save output
result.to_csv("combined_ddlr_catalogue.csv", index=False)