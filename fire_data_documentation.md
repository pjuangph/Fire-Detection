# FireSense HDF4 File Structure Documentation

## Overview

This document describes the structure and contents of HDF4 files produced by the NASA FireSense group. The file contains airborne sensor data including calibration parameters, geolocation information, and calibrated radiance measurements.

[Data Overview](https://asapdata.arc.nasa.gov/sensors/master/) - Description of data from FireSense


---

## File Dimensions

| Dimension | Size | Description |
|-----------|------|-------------|
| `NumberOfChannels` | 50 | Spectral channels/bands |
| `NumberOfScanlines` | 2736 | Along-track scan lines |
| `NumberOfPixels` | 716 | Cross-track pixels per scan |
| `NumberOfHeaderlines` | 150 | Header information lines |
| `MaxCharactersPerLine` | 97 | Maximum characters per header line |

---

## Dataset Descriptions

### 1. Spectral Channel Information

Datasets describing the spectral characteristics of each channel.

| Dataset | Dimensions | Shape | Description |
|---------|------------|-------|-------------|
| `Left50%ResponseWavelength` | (NumberOfChannels) | (50,) | Left wavelength at 50% spectral response |
| `Central100%ResponseWavelength` | (NumberOfChannels) | (50,) | Central wavelength at 100% spectral response |
| `Right50%ResponseWavelength` | (NumberOfChannels) | (50,) | Right wavelength at 50% spectral response |
| `EffectiveCentralWavelength_IR_bands` | (NumberOfChannels) | (50,) | Effective central wavelength for infrared bands |
| `SolarSpectralIrradiance` | (NumberOfChannels) | (50,) | Solar spectral irradiance per channel |

---

### 2. Temperature Correction Parameters

Calibration coefficients for temperature corrections.

| Dataset | Dimensions | Shape | Description |
|---------|------------|-------|-------------|
| `TemperatureCorrectionSlope` | (NumberOfChannels) | (50,) | Slope coefficient for temperature correction |
| `TemperatureCorrectionIntercept` | (NumberOfChannels) | (50,) | Intercept coefficient for temperature correction |

---

### 3. File Metadata

| Dataset | Dimensions | Shape | Description |
|---------|------------|-------|-------------|
| `DataSetHeader` | (NumberOfHeaderlines, MaxCharactersPerLine) | (150, 97) | File header containing metadata and processing information |

---

### 4. Scan Line Timing & Identification

Per-scanline temporal and identification data.

| Dataset | Dimensions | Shape | Description |
|---------|------------|-------|-------------|
| `ScanLineCounter` | (NumberOfScanlines) | (2736,) | Sequential scan line number |
| `GreenwichMeanTime` | (NumberOfScanlines) | (2736,) | GMT time of each scan line |
| `YearMonthDay` | (NumberOfScanlines) | (2736,) | Date of each scan line (YYYYMMDD format) |
| `ScanRate` | (NumberOfScanlines) | (2736,) | Scan rate for each line |
| `ScanlineTime` | (NumberOfScanlines) | (2736,) | Timestamp for each scan line |

---

### 5. Aircraft Position & Orientation

Navigation data for the aircraft platform.

| Dataset | Dimensions | Shape | Description |
|---------|------------|-------|-------------|
| `AircraftLatitude` | (NumberOfScanlines) | (2736,) | Aircraft latitude (degrees) |
| `AircraftLongitude` | (NumberOfScanlines) | (2736,) | Aircraft longitude (degrees) |
| `AircraftAltitude` | (NumberOfScanlines) | (2736,) | Aircraft altitude (meters) |
| `AircraftHeading` | (NumberOfScanlines) | (2736,) | Aircraft heading (degrees from north) |
| `AircraftPitch` | (NumberOfScanlines) | (2736,) | Aircraft pitch angle (degrees) |
| `AircraftRollCount` | (NumberOfScanlines) | (2736,) | Aircraft roll count/measurement |

---

### 6. Blackbody Calibration Data

Onboard blackbody reference measurements for radiometric calibration.

| Dataset | Dimensions | Shape | Description |
|---------|------------|-------|-------------|
| `BlackBody1Temperature` | (NumberOfScanlines) | (2736,) | Temperature of blackbody reference 1 (K) |
| `BlackBody2Temperature` | (NumberOfScanlines) | (2736,) | Temperature of blackbody reference 2 (K) |
| `TBack` | (NumberOfScanlines) | (2736,) | Background temperature |
| `BlackBody1Counts` | (NumberOfScanlines, NumberOfChannels) | (2736, 50) | Raw counts from blackbody 1 per channel |
| `BlackBody2Counts` | (NumberOfScanlines, NumberOfChannels) | (2736, 50) | Raw counts from blackbody 2 per channel |

---

### 7. Detector Head Measurements

Raw measurements from sensor heads.

| Dataset | Dimensions | Shape | Description |
|---------|------------|-------|-------------|
| `Head1Counts` | (NumberOfScanlines, NumberOfChannels) | (2736, 50) | Raw counts from detector head 1 |
| `Head2Counts` | (NumberOfScanlines, NumberOfChannels) | (2736, 50) | Raw counts from detector head 2 |

---

### 8. Analog Calibration Parameters

Per-scanline analog gain and offset values.

| Dataset | Dimensions | Shape | Description |
|---------|------------|-------|-------------|
| `AnalogGain` | (NumberOfScanlines, NumberOfChannels) | (2736, 50) | Analog gain per scan line and channel |
| `AnalogOffset` | (NumberOfScanlines, NumberOfChannels) | (2736, 50) | Analog offset per scan line and channel |

---

### 9. Radiometric Calibration Coefficients

Coefficients for converting raw counts to calibrated radiance.

| Dataset | Dimensions | Shape | Description |
|---------|------------|-------|-------------|
| `CalibrationSlope` | (NumberOfScanlines, NumberOfChannels) | (2736, 50) | Calibration slope (gain) |
| `CalibrationIntercept` | (NumberOfScanlines, NumberOfChannels) | (2736, 50) | Calibration intercept (offset) |

---

### 10. Pixel Geolocation

Geographic coordinates and elevation for each pixel.

| Dataset | Dimensions | Shape | Description |
|---------|------------|-------|-------------|
| `PixelLatitude` | (NumberOfScanlines, NumberOfPixels) | (2736, 716) | Latitude of each pixel (degrees) |
| `PixelLongitude` | (NumberOfScanlines, NumberOfPixels) | (2736, 716) | Longitude of each pixel (degrees) |
| `PixelElevation` | (NumberOfScanlines, NumberOfPixels) | (2736, 716) | Surface elevation at each pixel (meters) |

---

### 11. Viewing & Solar Geometry

Angular geometry for each pixel observation.

| Dataset | Dimensions | Shape | Description |
|---------|------------|-------|-------------|
| `SensorZenithAngle` | (NumberOfScanlines, NumberOfPixels) | (2736, 716) | Sensor zenith angle (degrees from nadir) |
| `SensorAzimuthAngle` | (NumberOfScanlines, NumberOfPixels) | (2736, 716) | Sensor azimuth angle (degrees from north) |
| `SolarZenithAngle` | (NumberOfScanlines, NumberOfPixels) | (2736, 716) | Solar zenith angle (degrees) |
| `SolarAzimuthAngle` | (NumberOfScanlines, NumberOfPixels) | (2736, 716) | Solar azimuth angle (degrees from north) |

---

### 12. Calibrated Science Data

| Dataset | Dimensions | Shape | Description |
|---------|------------|-------|-------------|
| `CalibratedData` | (NumberOfScanlines, NumberOfChannels, NumberOfPixels) | (2736, 50, 716) | Calibrated radiance data (primary science product) |

---