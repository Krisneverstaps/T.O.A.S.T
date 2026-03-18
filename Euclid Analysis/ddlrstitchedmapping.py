import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u

# CONFIGURATION 
CUTOUT_SIZE = 5.6 * u.arcsec  # High-zoom

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
catalog_path = (ROOT_DIR / "data" / "Euclid+data.csv").resolve()
df = pd.read_csv(catalog_path)

# List of IDs
ids = [
    "-519790213285648287", "-532257415282400330", "-536741865295643421",
    "-529462058278475857", "-532210123280315272", "-540305077278236258"
]

fig = plt.figure(figsize=(14, 9))
plt.subplots_adjust(wspace=0.01, hspace=0.01)  # Very tight together plots

plot_idx = 1  # track of successful plots to avoid gaps
legend_elements = []

for id_str in ids:
    fits_file = (ROOT_DIR / "data" / f"{id_str}.fits").resolve()
    if not fits_file.exists():
        continue

    # Match CSV Row
    row_match = df[df['Euclid_ID_x'].astype(str) == id_str]
    if row_match.empty:
        continue
    row = row_match.iloc[0]

    with fits.open(fits_file) as hdul:
        # Find Science Data
        science_hdu = next((h for h in hdul if h.data is not None and h.data.ndim == 2), None)
        if science_hdu is None:
            continue
        wcs = WCS(science_hdu.header, fobj=hdul)

        # Coordinates & Parameters
        sn_coords = SkyCoord(row["SN_RA"], row["SN_DEC"], unit="deg")
        ra0, dec0 = row["right_ascension"], row["declination"]
        pa, q, a = row["sersic_angle"], row["sersic_sersic_vis_axis_ratio"], row["sersic_sersic_vis_radius"]

        try:
            cutout = Cutout2D(
                science_hdu.data,
                sn_coords,
                size=CUTOUT_SIZE,
                wcs=wcs,
                mode='partial',
                fill_value=0
            )
        except Exception:
            print(f"Skipping {id_str}: Target is outside image bounds.")
            continue

        # ellipse calculation
        b = a * q
        t = np.linspace(0, 2*np.pi, 200)
        theta = np.radians(90 - pa)
        x_p, y_p = a * np.cos(t), b * np.sin(t)
        d_ra = (x_p * np.cos(theta) - y_p * np.sin(theta)) / 3600.0 / np.cos(np.radians(dec0))
        d_dec = (x_p * np.sin(theta) + y_p * np.cos(theta)) / 3600.0
        ra_el, dec_el = ra0 + d_ra, dec0 + d_dec

        # Plot
        ax = fig.add_subplot(2, 3, plot_idx, projection=cutout.wcs)

        # Scaling for dark background
        vmin, vmax = np.percentile(cutout.data, [10, 99.7])
        ax.imshow(cutout.data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, aspect='auto')

        # Overlays
        el_line, = ax.plot(ra_el, dec_el, transform=ax.get_transform('world'),
                           color='red', lw=1.8, label='Half-light ellipse ($R_e$)')
        sn_point = ax.scatter(row["SN_RA"], row["SN_DEC"], transform=ax.get_transform('world'),
                              color='blue', s=50, edgecolors='white', lw=0.6, label='Supernova')

        if not legend_elements:
            legend_elements = [el_line, sn_point]

        # Annotate image
        ax.text(0.05, 0.95, f"ID {id_str[-4:]}\nDDLR={row['DDLR']:.2f}",
                transform=ax.transAxes, color='white', fontsize=11,
                fontweight='bold', va='top', ha='left')

        # Remove all RA/Dec labels, ticks, and tick marks
        for coord in ax.coords:
            coord.set_axislabel('')            # Remove axis label
            coord.set_ticklabel_visible(False) # Remove tick numbers
            coord.set_ticks_visible(False)     # Remove tick marks

        plot_idx += 1
        if plot_idx > 6:
            break  # Grid full

# Legend at the top
if legend_elements:
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.94),
               ncol=2, frameon=False, fontsize=13)

plt.savefig("stitched_ddlr_figure.png", dpi=400, bbox_inches='tight')
plt.show()