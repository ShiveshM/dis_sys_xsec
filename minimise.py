#! /usr/bin/env python
"""
Parameter estimation for DIS xsec based on least squares method.
"""
from __future__ import print_function
import os

import numpy as np
import numpy.ma as ma
import h5py

from pisa.utils.log import logging, set_verbosity
from uncertainties import ufloat, unumpy
import ROOT

DIR = '/data/mandalia/NuTeV'
CACHE = DIR+'/cache.hdf5'
DATA_NU = '/data/icecube/data/NuTeV/nutev_data/data/nutevpack/NuVec.dat'
DATA_NUBAR= '/data/icecube/data/NuTeV/nutev_data/data/nutevpack/NubarVec.dat'
Y_BINNING = np.array([0, 0.001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                      0.9, 0.97, 1.0])
X_BINNING = np.array([0.0001, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                      0.6, 0.7, 0.8])
E_BINNING = np.array([30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200,
                      230, 260, 290, 320, 360])
Y_CENTERS = np.array([0.0005, 0.026, 0.075, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65,
                      0.75, 0.85, 0.935, 0.985])
X_CENTERS = np.array([0.015, 0.045, 0.08, 0.125, 0.175, 0.225, 0.275, 0.35,
                      0.45, 0.55, 0.65, 0.75])
E_CENTERS = np.array([35, 45, 55, 65, 75, 85, 95, 110, 130, 150, 170, 190, 215,
                      245, 275, 305, 340])
WA_NU = 0.677E-38
WA_NUBAR = 0.334E-38


def get_bin_sizes(bin_edges):
        return np.abs(np.diff(bin_edges))


def load_nutev_xsec_vec():
    """Load the NuTeV xsec data along with the total (stat+syst) error."""
    logging.info('Loading NuTeV data from {0} and {1}'.format(DATA_NU,
                                                              DATA_NUBAR))
    with open(DATA_NU, 'r') as f:
        table_nu = np.genfromtxt(f, usecols=range(9), skip_header=1)
    with open(DATA_NUBAR, 'r') as f:
        table_nubar = np.genfromtxt(f, usecols=range(9), skip_header=1)

    def decode_index(index):
        ybin = index % 100 - 1
        xbin = ((index % 10000) - ybin) / 100 - 1 - 1
        ebin = ((index % 1000000) - xbin*100 -ybin)/10000 - 1 - 1
        return [map(int, (ebin, xbin, ybin))]

    nu_decoded_index, nubar_decoded_index = [], []
    for index in table_nu[:,0]:
        nu_decoded_index.append(decode_index(index))
    for index in table_nubar[:,0]:
        nubar_decoded_index.append(decode_index(index))

    nu_decoded_index = np.vstack(nu_decoded_index)
    nubar_decoded_index = np.vstack(nubar_decoded_index)

    nu_sys = table_nu[:,3:].T
    nubar_sys = table_nubar[:,3:].T
    nu_sys_err_2 = nu_sys[0]**2 + nu_sys[1]**2 + nu_sys[2]**2 + \
            nu_sys[3]**2 + nu_sys[4]**2 + nu_sys[5]**2
    nubar_sys_err_2 = nubar_sys[0]**2 + nubar_sys[1]**2 + \
            nubar_sys[2]**2 + nubar_sys[3]**2 + nubar_sys[4]**2 + \
            nubar_sys[5]**2
    nu_xsec_array = unumpy.uarray(
        table_nu[:,1], np.sqrt(table_nu[:,2]**2 + nu_sys_err_2)
    )
    nubar_xsec_array = unumpy.uarray(
        table_nubar[:,1], np.sqrt(table_nubar[:,2]**2 + nubar_sys_err_2)
    )

    nu_data_matrix = np.zeros(map(len, (E_CENTERS, X_CENTERS, Y_CENTERS)))
    nubar_data_matrix = np.zeros(map(len, (E_CENTERS, X_CENTERS, Y_CENTERS)))
    nu_data_matrix = unumpy.uarray(nu_data_matrix, nu_data_matrix)
    nubar_data_matrix = unumpy.uarray(nubar_data_matrix, nubar_data_matrix)
    for idx, entry in enumerate(nu_decoded_index):
        nu_data_matrix[tuple(entry)] = nu_xsec_array[idx]
    for idx, entry in enumerate(nubar_decoded_index):
        nubar_data_matrix[tuple(entry)] = nubar_xsec_array[idx]
    nu_data_matrix = ma.masked_equal(nu_data_matrix, 0)
    nubar_data_matrix =  ma.masked_equal(nubar_data_matrix, 0)

    return nu_data_matrix, nubar_data_matrix


def load_genie_mc():
    """Load parameters from a GENIE gst ROOT file or from a cached HDF5
    file.
    """
    logging.info('Loading GENIE parameteters...')
    is_cache = False
    if os.path.isfile(CACHE):
        is_cache = True

    STR_R = ['pdg_array', 'energy_array', 'y_array', 'x_array', 'dis_array']
    def cache_events(params_dict):
        h5file = h5py.File(CACHE, 'w')
        for key in params_dict.iterkeys():
            h5file.create_dataset(key, data=params_dict[key])
        h5file.close()

    params_dict = {}
    if not is_cache:
        logging.info('Loading gst files from folder {0}'.format(DIR))
        infiles = []
        for file_name in os.listdir(DIR):
            file_path = DIR + '/' + file_name
            if os.path.isfile(file_path) and (file_path.endswith('.gst.root')):
                infiles.append(ROOT.TFile(file_path))
        for key in STR_R.iterkeys():
            params_dict[key] = []
        for r_file in infiles:
            for event in r_file.gst:
                params_dict['pdg_array'].append(event.neu)
                params_dict['energy_array'].append(event.Ev)
                params_dict['y_array'].append(event.y)
                params_dict['x_array'].append(event.x)
                params_dict['dis_array'].append(event.dis)
        for key in STR_R.iterkeys():
            params_dict[key] = np.array(params_dict[key])
        cache_events(params_dict) 
    else:
        logging.info('Loading from cached file {0}'.format(CACHE))
        h5file = h5py.File(CACHE, 'r')
        for key in h5file.iterkeys():
            params_dict[key] = np.array(h5file[key][:])
        h5file.close()

    return params_dict


def make_histo(array, weights):
    hist, edges = np.histogram(array, bins=Y_BINNING, weights=weights)
    hist_2, edges = np.histogram(array, bins=Y_BINNING, weights=weights**2)

    u_hist = unumpy.uarray(hist, np.sqrt(hist_2))
    return ma.masked_equal(u_hist, 0)


def evaluate(params_dict, x_systematic):
    e_bin_sizes = get_bin_sizes(E_BINNING).astype(float)
    x_bin_sizes = get_bin_sizes(X_BINNING).astype(float)
    y_bin_sizes = get_bin_sizes(Y_BINNING).astype(float)

    nu_histograms = unumpy.uarray(
        np.zeros(map(len, (e_bin_sizes, x_bin_sizes,
                           y_bin_sizes))).astype(float),
        np.zeros(map(len, (e_bin_sizes, x_bin_sizes,
                           y_bin_sizes))).astype(float)
    )
    nubar_histograms = unumpy.uarray(
        np.zeros(map(len, (e_bin_sizes, x_bin_sizes,
                           y_bin_sizes))).astype(float),
        np.zeros(map(len, (e_bin_sizes, x_bin_sizes,
                           y_bin_sizes))).astype(float)
    )
    scaling_nu, scaling_nubar = [], []

    logging.info('x_systematic {0}'.format(x_systematic))
    weights = np.exp(-params_dict['x_array']**x_systematic)

    nu_mask = params_dict['pdg_array'] > 0
    nubar_mask = params_dict['pdg_array'] < 0
    for e_idx, e_bin in enumerate(E_BINNING[:-1]):
        energy_array = params_dict['energy_array']
        energy_mask = (energy_array >= e_bin) & \
                (energy_array < E_BINNING[e_idx + 1])
        weights_e = weights[energy_mask]
        y_array_e = params_dict['y_array'][energy_mask]
        x_array_e = params_dict['x_array'][energy_mask]
        nu_mask_e = nu_mask[energy_mask]
        nubar_mask_e = nubar_mask[energy_mask]
        
        sigma_nu = sigma_nubar = 0
        nu_histograms[e_idx], nubar_histograms[e_idx] = {}, {}
        for x_idx, x_bin in enumerate(X_BINNING[:-1]):
            x_mask = (x_array_e >= x_bin) & (x_array_e < X_BINNING[x_idx+1])
            y_array_e_x = y_array_e[x_mask]
            weights_e_x = weights_e[x_mask]
            nu_mask_e_x = nu_mask_e[x_mask]
            nubar_mask_e_x = nubar_mask_e[x_mask]

            sigma_nu_x = make_histo(
                y_array_e_x[nu_mask_e_x], weights_e_x[nu_mask_e_x]
            )
            sigma_nu_x_e = sigma_nu_x * (1 / E_CENTERS[e_idx].astype(float))
            
            sigma_nubar_x = make_histo(
                y_array_e_x[nubar_mask_e_x], weights_e_x[nubar_mask_e_x]
            )
            sigma_nubar_x_e = sigma_nubar_x*(1/E_CENTERS[e_idx].astype(float))
            
            factors = 1 / (y_bin_sizes * x_bin_sizes[x_idx])
            nu_histograms[e_idx][x_idx] = sigma_nu_x_e * factors
            nubar_histograms[e_idx][x_idx] = sigma_nubar_x_e * factors
            sigma_nu += np.sum(unumpy.nominal_values(sigma_nu_x_e.data))
            sigma_nubar += np.sum(unumpy.nominal_values(sigma_nubar_x_e.data))

        scaling_nu.append((WA_NU / 1E-38) / sigma_nu)
        scaling_nubar.append((WA_NUBAR / 1E-38) / sigma_nubar)
    scaling_nu, scaling_nubar = map(ma.masked_invalid,
                                    (scaling_nu, scaling_nubar))
    nu_histograms = ma.masked_equal(nu_histograms, 0)
    nubar_histograms = ma.masked_equal(nubar_histograms, 0)
    logging.info('nu scaling parameters {0}'.format(scaling_nu))
    logging.info('nubar scaling parameters {0}'.format(scaling_nubar))

    scaled_nu_histograms = nu_histograms * \
            scaling_nu[:, np.newaxis][:, np.newaxis]
    scaled_nubar_histograms = nu_histograms * \
            scaling_nubar[:, np.newaxis][:, np.newaxis]

    return scaled_nu_histograms, scaled_nubar_histograms


# def calculate_chi2(data, mc):
#     numerator = np.power(unumpy.nominal_values(data) -
#                          unumpy.nominal_values(mc), 2)
#     print(numerator)
#     print(np.sum(numerator))

if __name__ == '__main__':
    set_verbosity(3)
    nu_data_matrix, nubar_data_matrix = load_nutev_xsec_vec()
    params_dict = load_genie_mc()
    NU_HISTOGRAMS, NUBAR_HISTOGRAMS = evaluate(params_dict, 0.2)
    # calculate_chi2(NU_HISTOGRAMS, nu_data_matrix)
