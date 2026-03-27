# -*- coding: utf-8 -*-
"""
Created March 23, 2026 by Karissa Jensen
"""

from pathlib import Path
import os
import numpy as np
from scipy.signal import find_peaks as find_peaks
from lmfit import Parameters, Minimizer 

#grabs group of 4 spectra and splits into wavenumber and data columns
#input: fileNm (file name of spectra set, output by pre-processing scripts)
#       directory (raw data folder, will automatically search for associated processed files)
#output: wns (single-column numpy array with wavenumbers)
#        allSpec (multi-column numpy array, each column one replicate spectrum, no wavenumber column)
def getSpecSet(fileNm, directory):
    file = np.loadtxt(directory / fileNm, delimiter=',')
    wns = file[:,0]
    allSpec = np.delete(file, 0, 1)
    return wns, allSpec




class ManageSpectrum:
    def __init__(self, x_all, y_all, spec_bl_points=None, low_x_bound=-np.inf, hi_x_bound=np.inf):
        # For FTIR data this is usually wavenumbers. 
        self.x_all = np.asarray(x_all, dtype=float)
        # The spectral data. 
        self.y_all = np.asarray(y_all, dtype=float)
        # Anchor points used to baseline correct data (optional). 
        self.spec_bl_points = np.asarray(spec_bl_points) if spec_bl_points is not None else None
        # User-input section boundaries - if none, it defaults to +/- inf which will just fit the whole spectrum.
        # BEWARE, though, this can easily be way too large and complex to fit with lmfit. 
        self.low_x_bound = low_x_bound
        self.hi_x_bound = hi_x_bound
    
    def get_section(self):
        # Attempts to pick a good index to slice the spectrum on with the given constraints.
        # First, check to see if there are input values to slice at for either boundary.
        if self.low_x_bound is not -np.inf or self.hi_x_bound is not np.inf: 
        # If so, first find spectral minima using scipy.signal's find_peaks on the inverse of the spectrum y array.
            minima_indices, _ = find_peaks(np.multiply(self.y_all, -1))
        # Then, check to see if the user provided a set of baseline anchor points to try (not required):
            if self.spec_bl_points is not None: 
                # Ffind the closest minimum to each baseline point and redefine minima_indices as a set of those minima only. 
                minima_indices = np.asarray([min(minima_indices, key=lambda x:abs(point-self.x_all[x])) for point in self.spec_bl_points])
        
        # now it checks each boundary individually and proceeds to set the slice index. 
        if self.low_x_bound is -np.inf: 
            low_x_index = min(np.indices(self.x_all.shape)[0])
        else: 
            # It will try to find low_x_index and hi_x_index based on the minima set above (whether user-defined or automatic).
            low_x_index = min(minima_indices, key=lambda x:abs(float(self.low_x_bound)-self.x_all[x]))
            # Sometimes the anchor points that work well for fittng a baseline don't work well for slicing.
            # In this case, this is handled by defaulting to the point in the whole spectrum closest to the user-defined bounds. 
            if abs(self.x_all[low_x_index] - float(self.low_x_bound)) > 20:                                           
                low_x_index = min(np.indices(self.x_all.shape)[0], key=lambda x:abs(float(self.low_x_bound)-self.x_all[x]))
        
        if self.hi_x_bound is np.inf: 
            hi_x_index = max(np.indices(self.x_all.shape)[0])
        else: 
            hi_x_index  = min(minima_indices, key=lambda x:abs(float(self.hi_x_bound)-self.x_all[x]))
            if abs(self.x_all[hi_x_index] - float(self.hi_x_bound)) > 20:                                       
                hi_x_index = min(np.indices(self.x_all.shape)[0], key=lambda x:abs(float(self.hi_x_bound)-self.x_all[x]))
        
        # Once indices are found, breaking into sections is simple. 
        x_section = self.x_all[low_x_index:hi_x_index+1]
        y_section = self.y_all[low_x_index:hi_x_index+1]
        return x_section, y_section
    
class BuildParameters:
    def __init__(self, params_file_path):
        self.params_file_path = Path(params_file_path)
    def params_table(self):
        # Obtains user-created parameters from specifically-formatted csv file. sets all empty spaces to nan. 
        return np.nan_to_num(np.genfromtxt(self.params_file_path, skip_header=1, delimiter=","), nan=np.nan) 
    def peak_count(self):
        return len(self.params_table()[:,0])
    def create_params_object(self):
        # part of lmfit package. See: https://lmfit.github.io/lmfit-py/parameters.html 
        # a dictionary of Parameters objects. 
        fit_params = Parameters() 

        # def mini_param_gen(suffix, index, vary_bool):
            # local function for parsing params_table into Parameter objects
        
        # the constraints for the peak area to height ratio are stored in a regular array.
        # lmfit encounters some problems with expression constraints, so they are handled differently within the fitting function. 
        ratio_constraints = self.params_table()[:, -3:]
    

# Local file management: finding files and extracting np array.
source_folder = Path("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723/chamber-5/20250110-0h")
filename = "20250113_PET-Ch5-Pos1-0h-Air_N723.csv"
# My (pre-processed) data is organized into files containing one wavenumber column and 4 replicate measurement columns
# getSpecSet retrieves all 4.
x_data_import, data_set = getSpecSet(filename, source_folder)

# Retrieves baseline anchor points file. The file contains only x-values. 
parent_directory = Path("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr")
blFileNm='PET-baseline-wns-fit.txt'
blFilePath = parent_directory / blFileNm      
baseline_points = np.loadtxt(blFilePath, delimiter=" ")
params_path = parent_directory / "peak-params_1517-1900-N723-lmfit.csv"

y_data_import = data_set[:,0]

# spectrum = ManageSpectrum(
#     x_data_import, y_data_import, 
#     spec_bl_points=baseline_points, 
#     low_x_bound=1517, hi_x_bound=1900)

build_params = BuildParameters(params_path)
print(build_params.peak_count())

