import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
fits_file = (ROOT_DIR / "data" / "EUC_VIS_SWL-DET-003025-02-1-0000000__20241017T053237.823256Z.fits").resolve()


# GALAXY PARAMETERS

ra0  = 53.22574151940563
dec0 = -28.24003306512169

pa_deg = -68.23918914794922
q      = 0.2958751320838928
R_vis  = 0.6445221304893494   # arcsec

sn_ra  = 53.225521
sn_dec = -28.239893

target = SkyCoord(ra0, dec0, unit="deg")

hdul = fits.open(fits_file, memmap=True)

science_hdu = None

for hdu in hdul:
    if hdu.name.endswith("SCI"):
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



wcs = WCS(science_hdu.header)
size = 30 * u.arcsec

cutout = Cutout2D(science_hdu.data, target, size, wcs=wcs)


# a = R_vis              
a = R_vis / np.sqrt(q)
b = a * q

theta = np.radians(pa_deg)
t = np.linspace(0, 2*np.pi, 500)

x_prime = a * np.cos(t)
y_prime = b * np.sin(t)

delta_ra  = x_prime * np.sin(theta) + y_prime * np.cos(theta)
delta_dec = x_prime * np.cos(theta) - y_prime * np.sin(theta)

ra_ellipse  = ra0 + delta_ra / (3600 * np.cos(np.radians(dec0)))
dec_ellipse = dec0 + delta_dec / 3600


# PLOT
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection=cutout.wcs)

vmin, vmax = np.percentile(cutout.data, [5, 99])
ax.imshow(cutout.data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

ax.plot(
    ra_ellipse,
    dec_ellipse,
    transform=ax.get_transform('world'),
    color='red',
    linewidth=2,
    label='Half-light ellipse'
)

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