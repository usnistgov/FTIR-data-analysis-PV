# -*- coding: utf-8 -*-
"""
Created March 23, 2026 by Karissa Jensen
"""

from pathlib import Path
import os
import numpy as np
from scipy.signal import find_peaks as find_peaks
from lmfit import Parameters, Minimizer 
import math
import time 
import colorsys
import matplotlib.pyplot as plt

"""
TO-DO
-remove references to wavenumbers and replace with generic x data label
-some class functions could be attributes assigned under __init__ instead of methods - cleaner 
"""

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
    
    def get_section_indices(self):
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
        return low_x_index, hi_x_index
    @property
    def x_section(self):   
    # Once indices are found, breaking into sections is simple.         
        return self.x_all[self.get_section_indices()[0]:self.get_section_indices()[1]+1]
    @property
    def y_section(self):    
        return self.y_all[self.get_section_indices()[0]:self.get_section_indices()[1]+1]
    
class BuildParameters:
    def __init__(self, params_file_path):
        self.params_file_path = Path(params_file_path)
        # Obtains user-created parameters from specifically-formatted csv file. sets all empty spaces to nan. 
        self.params_table = np.nan_to_num(np.genfromtxt(self.params_file_path, skip_header=1, delimiter=","), nan=np.nan) 
        self.peak_count = len(self.params_table[:,0])
        # the constraints for the peak area to height ratio are stored in a regular array.
        # lmfit encounters some problems with expression constraints, so they are handled differently within the fitting function. 
        self.ratio_constraints = self.params_table[:, -3:]
        self.peak_dict = {i : str(self.params_table[:,1][i].astype(int)) for i in range(len(self.params_table[:,1]))}
    
    def params_object(self):
        # part of lmfit package. See: https://lmfit.github.io/lmfit-py/parameters.html 
        # a dictionary of Parameters objects. 
        fit_params = Parameters() 

        def mini_param_generator(suffix, index, vary_bool):
            # local function for parsing params_table into Parameter objects. 
            # Fills with None, -inf, or inf if no value provided.
            fit_params.add(name = f"p{pk_ID}_{suffix}", vary=vary_bool,
                           value=self.params_table[i, index] if np.isnan(self.params_table[i,index]) == False else None,
                           min=self.params_table[i, index-1] if np.isnan(self.params_table[i, index-1]) == False else -np.inf, 
                           max=self.params_table[i, index+1] if np.isnan(self.params_table[i, index+1]) == False else np.inf)

        for i in range(self.peak_count):
            pk_ID = i+1
            # parse the parameters table file for constraints and add them to the Parameter dictionary 
            mini_param_generator('wn', 1, True)
            mini_param_generator('area', 4, True)
            mini_param_generator('fwhm', 7, True)
            mini_param_generator('c', 10, False)
            # bug fit for area = 0 causing height = 0 which then causes zero division error in expression params. 
            if fit_params[f"p{pk_ID}_area"].min==0:
                fit_params[f"p{pk_ID}_area"].set(min=0.0001) 
        
            #add derived parameters: height (h) and area/height ratio (ratio)
            fit_params.add(f"p{pk_ID}_h", vary=False)
            fit_params[f"p{pk_ID}_h"].expr = f"p{pk_ID}_area / ( (p{pk_ID}_fwhm / (2*sqrt(2*log(2)))) *sqrt(2*pi))"
        
            fit_params.add(f"p{pk_ID}_ratio", vary=False)
            fit_params[f"p{pk_ID}_ratio"].expr = f"p{pk_ID}_area / p{pk_ID}_h"
        return fit_params

    # def peak_dict(self):
    #     return {i : str(self.params_table[:,1][i].astype(int)) for i in range(len(self.params_table[:,1]))}
"""
post processing also needs better commenting 
"""
class PostProcessing:
    def __init__(self, result_params, x_section, y_section, best_fit, x_label, plot_title, peak_dict, peak_count):
        # results parameter from sharpness_fit
        self.result_params = result_params
        # user input label for x data (wavenumbers for FTIR)
        self.x_label = str(x_label) 
        self.plot_title = str(plot_title)
        self.peak_dict = peak_dict 
        self.x_section = x_section
        self.y_section = y_section
        # self.columns_list = [self.x_label, 'area', 'FWHM', 'y int', 'height']
        self.best_fit = best_fit
        self.param_vals = self.result_params.valuesdict()
    
    def params_array(self):
        # parsing results Parameters dictionary into human-readable/exportable array 
        wns, areas, fwhms, cs, heights = [], [], [], [], []
        for param in self.result_params.valuesdict():
            param_type = param.split("_")[1]
            match param_type:
                case 'wn': wns.append(self.param_vals[param])
                case 'area': areas.append(self.param_vals[param])
                case 'fwhm': fwhms.append(self.param_vals[param])
                case 'c': cs.append(self.param_vals[param])
                case 'h': heights.append(self.param_vals[param])
        return np.array([wns, areas, fwhms, cs, heights]).T
    
    @property
    def plot_colors(self):
        # color generation: https://stackoverflow.com/questions/876853/generating-color-ranges-in-python 
        hls_tups = [(x/self.peak_count, 0.4, 1) for x in range(self.peak_count)]
        rgb_tups = list(map(lambda x: colorsys.hls_to_rgb(*x), hls_tups))
        return rgb_tups
    
    def plot_fit(self):
        # result_params, best_fit x_section, y_section, 
        figure, fit_plot = plt.subplots(figsize=(10, 5))
        plt.title(f"{self.plot_title}")
        # plot original input spectrum 
        fit_plot.plot(self.x_section, self.y_section, c='black', label='spectrum', linewidth=0.5)
        # plot full fit spectrum on top in red 
        fit_plot.plot(self.x_section, self.best_fit, c='red', label='fit', linewidth=0.5)
        for i in range(self.peak_count):
            pk_ID = i+1
            h = self.param_vals[f'p{pk_ID}_h'] 
            ratio = self.param_vals[f'p{pk_ID}_ratio'] 
            # input final parameter values into function to create one peak 
            single_peak = gauss_area(self.x_section, 
                                      self.param_vals[f'p{pk_ID}_wn'],
                                      self.param_vals[f'p{pk_ID}_area'],    
                                      self.param_vals[f'p{pk_ID}_fwhm'],
                                      self.param_vals[f'p{pk_ID}_c'])
            fit_plot.plot(self.x_section, single_peak, c=self.plot_colors[i], linewidth=0.5)

# custom gaussian function, though lmfit has built in models for gaussian, lorentzian, voight, etc that could be implemented if desired. 
def gauss_area(x, wn, area, fwhm, c):
    return c + area/(fwhm*math.sqrt(math.pi/(4*math.log(2)))) * np.exp(-4*math.log(2)*(x-wn)**2/(fwhm**2)) 

# the main course
def sharpness_fit(x_section, y_section, params, ratio_constraints, peak_count, multiplier):
    def residual(pars, x, data, peak_count):
        # local function for minimizer 
        # unpacking parameters dictionary at top of function per: https://lmfit.github.io/lmfit-py/fitting.html 
        param_vals = pars.valuesdict()
        # building model
        model = np.zeros_like(x)
        # this uses a penalty method to enforce constraints that lmfit otherwise does not prioritize
        # (constraints on expressions)
        total_penalty = []
        # penalty scale is based on max value of y data. Set multiplier through testing. 
        penalty_scale = np.max(data) * multiplier


        for i in range(peak_count):
            pk_ID = i+1 
            # collect individual parameters by name from Parameters dictionary 
            wn, area, fwhm, c = param_vals[f'p{pk_ID}_wn'], param_vals[f'p{pk_ID}_area'], param_vals[f'p{pk_ID}_fwhm'], param_vals[f'p{pk_ID}_c']

            # penalty method used to prevent dependent parameters being ignored. 
            # get min and max from array pulled from csv: 
            ratio_min, ratio_max = ratio_constraints[i, 0], ratio_constraints[i, 2]
            # get value being held in parameter at current iteration from unpacked dictionary
            ratio_val = param_vals[f'p{pk_ID}_ratio']
            

            """this section needs comments edited
            here to end of sharpness_fit function"""
            # using np.maximum reduces number of if statements in loop and leaves penalties "vectorized" according to gemini which is supposed to be faster
            if np.isfinite(ratio_max):   
                total_penalty.append(np.maximum(0, ratio_val - ratio_max)**2 * penalty_scale)   
            if np.isfinite(ratio_min):
                total_penalty.append(np.maximum(0, ratio_min - ratio_val)**2 * penalty_scale)

            #add peak to model (external function)
            model += gauss_area(x, wn, area, fwhm, c)

        return np.concatenate([(model - data), total_penalty])  #concatenate allegedly runs faster here than regular add. 
    
    # see lmfit documentation - standard implementation. 
    mini = Minimizer(residual, params, fcn_args=(x_section, y_section, peak_count))
    out = mini.minimize(method='leastsq')     #leastsq: Levenberg-Marquardt (default)
    
    # removes the penalty values at the end of the residual array for the final output residual
    clean_residual = out.residual[:len(y_section)]
    best_fit = y_section + clean_residual
    result_params = out.params 
    return result_params, best_fit

"""
-------------------------------------------------------------------------------
"""

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


file_start_time = time.perf_counter()

spectrum = ManageSpectrum(x_data_import, y_data_import, 
                          spec_bl_points=baseline_points, 
                          low_x_bound=1517, hi_x_bound=1900)

build_params = BuildParameters(params_path)


output_parameters, section_fit = sharpness_fit(spectrum.x_section, spectrum.y_section, 
                                               build_params.params_object(), build_params.ratio_constraints, 
                                               build_params.peak_count, 0)

print(build_params.params_object().pretty_print())
results = PostProcessing(output_parameters, spectrum.x_section, spectrum.y_section, section_fit,
                         'wavenumbers', filename, 
                         build_params.peak_dict, build_params.peak_count)


file_time_seconds = time.perf_counter() - file_start_time
file_time_stamp = f"fit time: {round(file_time_seconds, 2)} seconds"

results.plot_fit()

plot_time_seconds = time.perf_counter() - file_start_time - file_time_seconds
plot_time_stamp = f"plot time: {round(plot_time_seconds, 2)} seconds"
print(file_time_stamp)
print(plot_time_stamp)

# time stamping 


# elapsed_seconds = file_end_time - file_start_time 
# elapsed_minutes = elapsed_seconds / 60 
# time_stamp = f"{round(elapsed_seconds, 2)} seconds"
# print(f"processing time: {time_stamp}")
