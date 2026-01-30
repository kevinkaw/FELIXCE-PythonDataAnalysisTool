![FELIXCE logo](./logo/logo_FELIXCE_solid.png)

## Overview

The FELIXCE Python Data Analysis Tool is a software package designed for automated processing and analysis of infrared photodissociation spectroscopy data from the FELIX (Free Electron Laser for Infrared eXperiments) laboratory. This tool enables researchers to efficiently analyze mass spectrometry data to determine infrared yield spectra by comparing molecular ion signal intensities with and without IR radiation, thereby streamlining the data analysis process allowing researchers to focus on scientific interpretation rather than data processing challenges.

## Key Features

#### 🔬 Comprehensive Data Processing
- **HDF5 Data Import:** Native support for FELIX laboratory data formats
- **Batch Processing:** Simultaneous analysis of multiple experimental scans
- **Wavenumber Standardization:** Automatic organization and rounding for consistent data handling
- **Multi-file Integration:** Seamless combination of data across experimental sessions

#### 📊 Spectroscopic Analysis
- **IR Yield Calculations:** Automated calculation of depletion and IR yields
- **Baseline Correction:** Normalization using user-defined reference regions
- **Peak Optimization:** Automatic detection and integration of molecular ion signals
- **Isotope Analysis:** Multi-isotope pattern recognition and quantification

#### 🖥️ User-Friendly Interface
- **Streamlit Web GUI:** Interactive, browser-based analysis workflow
- **Real-time Visualization**: Dynamic plotting of mass spectra and IR yield data
- **Parameter Configuration**: Easy adjustment of analysis settings through intuitive forms
- **Progress Tracking:** Step-by-step workflow guidance with visual feedback

#### 🧮 Molecular Complex Support
- **Theoretical Mass Calculations:** Automated computation for user-defined molecular formulas
- **Charge State Handling:** Support for various ionization states (e.g., Fe5⁺-Ar1)
- **Flexible Peak Detection:** Customizable mass windows and integration parameters

## Data Workflow

1. 📥 **Import:** Load FELIX HDF5 files containing raw IR photodissociation measurements
2. ⚙️ **Process:** Extract and organize wavenumber-specific mass spectral data
3. 📈 **Baseline Correct:** Normalize spectra using reference mass regions for accurate quantification
4. 🔍 **Analyze:** Calculate IR yields by comparing signal intensities with/without IR irradiation
5. 📊 **Visualize & Export:** Generate IR spectra and export results

## Output Formats

- **IR Yield Spectra:** Complete photodissociation efficiency vs. wavenumber datasets (CSV)
- **Mass Spectra:** Baseline-corrected spectral data at specific IR frequencies
- **Analysis Tables:** Comprehensive data compilations for further research
- **Visualization Files:** HTML tables and plots for data inspection and presentation