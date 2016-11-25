#! /usr/bin/env python
"""
Testing correlation of dis parameters
"""
from __future__ import print_function
import os
from itertools import product

import numpy as np
import numpy.ma as ma
from numpy.linalg import inv
import scipy.optimize as opt
import h5py

from pisa.utils.log import logging, set_verbosity
from uncertainties import ufloat, unumpy
import ROOT

import matplotlib as mpl
# headless mode
mpl.use('Agg')
# fonts
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

TYPE = 14
NAME = r'gevgen_{0}'.format(TYPE)
CACHE = '/data/icecube/data/NuTeV/{0}/cache.hdf5'.format(TYPE)
DATA_NU = '/data/icecube/data/NuTeV/nutev_data/data/nutevpack/NuVec.dat'
DATA_NUBAR= '/data/icecube/data/NuTeV/nutev_data/data/nutevpack/NubarVec.dat'
DATA_NU_COV = '/data/icecube/data/NuTeV/nutev_data/data/nutevpack/NuMatr.hdf5'
DATA_NUBAR_COV = '/data/icecube/data/NuTeV/nutev_data/data/nutevpack/NubarMatr.hdf5'
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


def load_genie_mc():
    """Load parameters from a GENIE gst ROOT file or from a cached HDF5
    file if it exists.
    """
    logging.info('Loading GENIE parameteters...')
    is_cache = False
    if os.path.isfile(CACHE):
        is_cache = True

    STR_R = ['pdg_array', 'energy_array', 'y_array', 'x_array', 'dis_array',
             'cc_array']
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
                params_dict['cc_array'].append(event.cc)
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


def only_discc(params_dict):
    x_array = params_dict['x_array']
    invalid_entries = x_array < 0
    discc = (params_dict['dis_array'] & params_dict['cc_array']).astype(bool)
    params_dict['x_array'] = x_array[~invalid_entries & discc]
    params_dict['energy_array'] = \
            params_dict['energy_array'][~invalid_entries & discc]
    params_dict['y_array'] = \
            params_dict['y_array'][~invalid_entries & discc]
    params_dict['pdg_array'] = \
            params_dict['pdg_array'][~invalid_entries & discc]
    params_dict['dis_array'] = \
            params_dict['dis_array'][~invalid_entries & discc]
    params_dict['cc_array'] = \
            params_dict['cc_array'][~invalid_entries & discc]

    return params_dict

def make_histo(array, weights):
    hist, edges = np.histogram(array, bins=Y_BINNING, weights=weights)
    hist_2, edges = np.histogram(array, bins=Y_BINNING, weights=weights**2)

    u_hist = unumpy.uarray(hist, np.sqrt(hist_2))
    return ma.masked_equal(u_hist, 0)


def evaluate(params_dict, systematics, nu=True, shape_only=False):
    a_sys = systematics
    # a_sys, b_sys, c_sys = systematics
    e_bin_sizes = get_bin_sizes(E_BINNING).astype(float)
    x_bin_sizes = get_bin_sizes(X_BINNING).astype(float)
    y_bin_sizes = get_bin_sizes(Y_BINNING).astype(float)

    nu_histograms = unumpy.uarray(
        np.zeros(map(len, (e_bin_sizes, x_bin_sizes,
                           y_bin_sizes))).astype(float),
        np.zeros(map(len, (e_bin_sizes, x_bin_sizes,
                           y_bin_sizes))).astype(float)
    )
    scaling = []

    logging.info('a_systematic {0}'.format(a_sys))
    # logging.info('b_systematic {0}'.format(b_sys))
    # logging.info('c_systematic {0}'.format(c_sys))
    # weights = c_sys + np.power(params_dict['x_array'], -a_sys) * b_sys
    weights = np.power(params_dict['x_array'], -a_sys)

    if nu: nu_mask = params_dict['pdg_array'] > 0
    else: nu_mask = params_dict['pdg_array'] < 0
    for e_idx, e_bin in enumerate(E_BINNING[:-1]):
        energy_array = params_dict['energy_array']
        energy_mask = (energy_array >= e_bin) & \
                (energy_array < E_BINNING[e_idx + 1])
        weights_e = weights[energy_mask]
        y_array_e = params_dict['y_array'][energy_mask]
        x_array_e = params_dict['x_array'][energy_mask]
        nu_mask_e = nu_mask[energy_mask]
        
        sigma_nu  = 0
        nu_histograms[e_idx] = {}
        for x_idx, x_bin in enumerate(X_BINNING[:-1]):
            x_mask = (x_array_e >= x_bin) & (x_array_e < X_BINNING[x_idx+1])
            y_array_e_x = y_array_e[x_mask]
            weights_e_x = weights_e[x_mask]
            nu_mask_e_x = nu_mask_e[x_mask]

            sigma_nu_x_nw = make_histo(
                y_array_e_x[nu_mask_e_x], np.ones(weights_e_x[nu_mask_e_x].shape)
            )
            sigma_nu_x = make_histo(
                y_array_e_x[nu_mask_e_x], weights_e_x[nu_mask_e_x]
            )
            sigma_nu_x_e_nw = sigma_nu_x_nw * (1 / E_CENTERS[e_idx].astype(float))
            sigma_nu_x_e = sigma_nu_x * (1 / E_CENTERS[e_idx].astype(float))
            
            factors = 1 / (y_bin_sizes * x_bin_sizes[x_idx])
            nu_histograms[e_idx][x_idx] = sigma_nu_x_e * factors
            if shape_only:
                sigma_nu += np.sum(unumpy.nominal_values(sigma_nu_x_e.data))
            else:
                sigma_nu += np.sum(unumpy.nominal_values(sigma_nu_x_e_nw.data))

        if nu: scaling.append((WA_NU / 1E-38) / sigma_nu)
        else: scaling.append((WA_NUBAR / 1E-38) / sigma_nu)
    scaling = ma.masked_invalid(scaling)
    nu_histograms = ma.masked_equal(nu_histograms, 0)

    scaled_histograms = nu_histograms * \
            scaling[:, np.newaxis][:, np.newaxis]
    return scaled_histograms


if __name__ == '__main__':
    set_verbosity(3)

    params_dict = load_genie_mc()
    params_dict = only_discc(params_dict)

    a_vals = np.linspace(0.01, 0.2, 40)

    nu = True
    # nu = False

    expectation_array = []
    b_vals_array = []
    for a in a_vals:
        if nu:
            opt_histogram = evaluate(
                params_dict, a, nu=True, shape_only=False
            )
        else:
            opt_histogram = evaluate(
                params_dict, a, nu=False, shape_only=False
            )

        if nu: expectation = 1 - 1.65125 * a
        else: expectation = 1 - 1.8073 * a
        expectation_array.append(expectation)

        x_bin_sizes = get_bin_sizes(X_BINNING).astype(float)
        y_bin_sizes = get_bin_sizes(Y_BINNING).astype(float)
        b_vals = []
        for e_idx, e_bin in enumerate(E_CENTERS):
            integral = 0
            for x_idx, x_bin in enumerate(X_CENTERS):
                integral += np.sum(opt_histogram[e_idx][x_idx].data * \
                                   y_bin_sizes) * x_bin_sizes[x_idx]
            # print(integral)
            if nu: b = (WA_NU / 1E-38) / integral
            else: b = (WA_NUBAR / 1E-38) / integral
            b_vals.append(unumpy.nominal_values(b))
        b_val = ufloat(np.mean(b_vals), np.std(b_vals))
        b_vals_array.append(b_val)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.set_xlim(np.min(a_vals)-0.02, np.max(a_vals)+0.02)
    ax.set_ylim(0.6, 1.0)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel('a', fontsize=18)
    ax.set_ylabel('b', fontsize=15)
    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y=ymaj, ls=':', color='gray', alpha=0.7, linewidth=1)
    for xmaj in ax.xaxis.get_majorticklocs():
        ax.axvline(x=xmaj, ls=':', color='gray', alpha=0.7, linewidth=1)

    ax.errorbar(a_vals, expectation_array, xerr=0, yerr=0, capsize=3,
                alpha=0.5, linestyle='--', markersize=2, linewidth=1,
                color='blue', label='expectation')
    ax.errorbar(a_vals, unumpy.nominal_values(b_vals_array), xerr=0,
                yerr=unumpy.std_devs(b_vals_array), capsize=3, alpha=0.5,
                linestyle='--', markersize=2, linewidth=1, color='red',
                label='calculated')
    legend = ax.legend(prop=dict(size=12))
    plt.setp(legend.get_title(), fontsize=18)
    if nu:
        out = 'correlation_nu.png'
        tex = r'$\nu$'
    else:
        out = 'correlation_nubar.png'
        tex = r'$\bar{\nu}$'
    at = AnchoredText(tex, prop=dict(size=20), frameon=True, loc=2)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.5")
    ax.add_artist(at)
    fig.savefig(out, bbox_inches='tight', dpi=150)
