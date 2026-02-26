import numpy as np
from scipy.stats import ttest_ind, ks_2samp
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

ARCSEC_PER_DEG = 3600.0

def compute_dlr_vis_row(row):
    # Positions
    host_ra = row["right_ascension"] # Euclids value
    host_dec = row["declination"] # Euclids value

    sn_ra = row["SN_RA"]
    sn_dec = row["SN_DEC"]

    # parameters
    circ = row["sersic_sersic_vis_radius"]
    q = row["sersic_sersic_vis_axis_ratio"]      # axis ratio b/a
    pa_deg = row["sersic_angle"]                 # degrees (East of North)
    a = circ / np.sqrt(q)          # semi-major axis
    # If invalid, returns NaN
    if pd.isna(a) or pd.isna(q) or a <= 0 or q <= 0:
        return np.nan


    # Convert coordinate offsets to arcsec
    delta_ra = (sn_ra - host_ra) * np.cos(np.radians(host_dec)) * ARCSEC_PER_DEG
    delta_dec = (sn_dec - host_dec) * ARCSEC_PER_DEG

    # Rotate into galaxy frame
    theta = np.radians(pa_deg)

    x_prime = delta_ra * np.sin(theta) + delta_dec * np.cos(theta)
    y_prime = delta_ra * np.cos(theta) - delta_dec * np.sin(theta)

    # Semi-minor axis
    b = a * q

    # Elliptical normalised radius
    d_DLR = np.sqrt((x_prime / a)**2 + (y_prime / b)**2)

    angular_sep = np.sqrt(delta_ra**2 + delta_dec**2)

    DDLR = angular_sep / d_DLR

    return pd.Series({'DDLR': DDLR, 'd_DLR': d_DLR, 'ANGSEP': angular_sep})

# Apply to full data
def add_dlr_columns(df):
    results = df.apply(compute_dlr_vis_row, axis=1)
    df = pd.concat([df, results], axis=1)
    return df



if __name__ == "__main__":
    euclid_morph_path = (ROOT_DIR / "data" / "Morphology_With_SN_Coords.csv").resolve()

# LOAD EUCLID DATA
    df = pd.read_csv(euclid_morph_path)

    df = add_dlr_columns(df)

    print(df[["object_id", "DDLR"]].head())

    df.to_csv("Euclid+data.csv", index=False)