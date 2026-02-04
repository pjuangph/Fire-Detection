import glob
import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC

# Pick the first HDF file to inspect
hdf_files = sorted(glob.glob('ignite_fire_data/*.hdf'))
print(f"Found {len(hdf_files)} HDF files\n")

filepath = hdf_files[0]
f = SD(filepath, SDC.READ)

# Print all dataset names and shapes
print(f"File: {filepath}")
print(f"Global attrs: {list(f.attributes().keys())}\n")
print("Datasets:")
for name, (dims, shape, dtype, nattrs) in sorted(f.datasets().items()):
    print(f"  {name:40s} shape={shape}")

# Read key datasets
cal_data = f.select('CalibratedData')
scale_factors = cal_data.attributes()['scale_factor']
raw = cal_data[:].astype(np.float32)
# Apply scale factors per channel
for ch in range(raw.shape[1]):
    raw[:, ch, :] *= scale_factors[ch]
radiance = raw  # (scanlines, 50 channels, 716 pixels)

lat = f.select('PixelLatitude')[:]
lon = f.select('PixelLongitude')[:]
wavelengths = f.select('Central100%ResponseWavelength')[:]

fill_value = -999.0
radiance[radiance < 0] = np.nan
lat[lat == fill_value] = np.nan
lon[lon == fill_value] = np.nan

f.end()

print(f"\nRadiance shape: {radiance.shape}  (scanlines, channels, pixels)")
print(f"Wavelengths (μm): {wavelengths}")

# --- Plot ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(filepath.split('/')[-1], fontsize=13)

# Pick channels across the spectrum: VNIR, SWIR, TIR
channels = [3, 15, 25, 35, 44, 49]
labels = ['VNIR', 'VNIR/SWIR', 'SWIR', 'MWIR', 'TIR', 'TIR']

for ax, ch, label in zip(axes.flat, channels, labels):
    img = radiance[:, ch, :]
    vmin, vmax = np.nanpercentile(img, [2, 98])
    im = ax.imshow(img, aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
    ax.set_title(f"Ch {ch+1} ({wavelengths[ch]:.2f} μm) - {label}")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Scanline")
    plt.colorbar(im, ax=ax, label="W/m²/sr/μm", fraction=0.046)

plt.tight_layout()
plt.savefig("radiance_overview.png", dpi=150)
plt.show()

# Geographic extent plot
fig2, ax2 = plt.subplots(figsize=(8, 8))
# Use a thermal channel for fire detection context
thermal_ch = 44  # ~11 μm TIR
img = radiance[:, thermal_ch, :]
vmin, vmax = np.nanpercentile(img, [2, 98])
sc = ax2.pcolormesh(lon, lat, img, cmap='inferno', vmin=vmin, vmax=vmax, shading='auto')
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
ax2.set_title(f"Ch {thermal_ch+1} ({wavelengths[thermal_ch]:.2f} μm) - Georeferenced")
ax2.set_aspect('equal')
plt.colorbar(sc, ax=ax2, label="W/m²/sr/μm")
plt.tight_layout()
plt.savefig("georeferenced_thermal.png", dpi=150)
plt.show()
