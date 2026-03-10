import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u


# PATHS
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

fits_file = (ROOT_DIR / "data" /
             "60095881.60095881-16.IRAC.1.mosaic (1).fits").resolve()

catalog_path = (ROOT_DIR / "data" / "Euclid+data.csv").resolve()

df = pd.read_csv(catalog_path)
print(df[["Euclid_ID_x", "SN_RA", "SN_DEC", "DDLR", "right_ascension", "declination" ]].head())

sn_ra_target = 53.471634
sn_dec_target = -29.379313
   
# Match SN
match = df.loc[
    (np.isclose(df["SN_RA"], sn_ra_target, atol=1e-6)) &
    (np.isclose(df["SN_DEC"], sn_dec_target, atol=1e-6))
]

if len(match) == 0:
    raise ValueError("No matching SN found in catalogue.")

row = match.iloc[0]



# EXTRACT GALAXY PARAMETERS

ra0 = row["right_ascension"]
dec0 = row["declination"]

pa_deg = row["sersic_angle"]
q = row["sersic_sersic_vis_axis_ratio"]
R_vis = row["sersic_sersic_vis_radius"]

sn_ra = row["SN_RA"]
sn_dec = row["SN_DEC"]

# Basic validation
if np.isnan(R_vis) or np.isnan(q) or q <= 0 or R_vis <= 0:
    raise ValueError("Invalid morphology parameters.")


# FIND CORRECT SCI EXTENSION

target = SkyCoord(ra0, dec0, unit="deg")

science_hdu = None

with fits.open(fits_file, memmap=True) as hdul:

    for hdu in hdul:
        if hdu.name.endswith("SCI") and hdu.data is not None:
            try:
                wcs = WCS(hdu.header)
                px, py = wcs.world_to_pixel(target)

                if (0 <= px < hdu.data.shape[1]) and (0 <= py < hdu.data.shape[0]):
                    science_hdu = hdu
                    print("Found galaxy in extension:", hdu.name)
                    break

            except Exception:
                continue

    if science_hdu is None:
        raise RuntimeError("Galaxy not found in any SCI extension.")

    # CREATE CUTOUT

    wcs = WCS(science_hdu.header)
    size = 30 * u.arcsec

    cutout = Cutout2D(science_hdu.data, target, size, wcs=wcs)



# BUILD HALF-LIGHT ELLIPSE

# Interpret R_vis as semi-major axis
a = R_vis
# circularised instead, use:
# a = R_vis / np.sqrt(q)

b = a * q

theta = np.radians(pa_deg)
t = np.linspace(0, 2*np.pi, 500)

# Ellipse in galaxy frame
x_prime = a * np.cos(t)
y_prime = b * np.sin(t)

# Rotate back to sky frame
delta_ra = x_prime * np.sin(theta) + y_prime * np.cos(theta)
delta_dec = x_prime * np.cos(theta) - y_prime * np.sin(theta)

# Convert arcsec offsets to degrees
ra_ellipse = ra0 + delta_ra / (3600.0 * np.cos(np.radians(dec0)))
dec_ellipse = dec0 + delta_dec / 3600.0


# PLOT

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection=cutout.wcs)

vmin, vmax = np.percentile(cutout.data, [5, 99])

ax.imshow(
    cutout.data,
    origin="lower",
    cmap="gray",
    vmin=vmin,
    vmax=vmax
)

# Half-light ellipse
ax.plot(
    ra_ellipse,
    dec_ellipse,
    transform=ax.get_transform('world'),
    color='red',
    linewidth=2,
    label='Half-light ellipse (Re)'
)

# Supernova position
ax.scatter(
    sn_ra,
    sn_dec,
    transform=ax.get_transform('world'),
    color='blue',
    s=60,
    label='Supernova'
)

ax.set_xlabel("RA")
ax.set_ylabel("Dec")
ax.legend()

plt.tight_layout()
plt.show()