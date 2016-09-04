#! /usr/bin/env python
"""
Parameter estimation for DIS xsec based on least squares method.
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


def load_nutev_corrmatrix():
    """Load the inverse correlation matrix from a HDF5 file."""
    logging.info('Loading correlation matrix from {0} and '
                 '{1}'.format(DATA_NU_COV, DATA_NUBAR_COV))
    h5file_nu = h5py.File(DATA_NU_COV, 'r')
    h5file_nubar = h5py.File(DATA_NUBAR_COV, 'r')
    inv_cov_table_nu = np.array(h5file_nu['NuMatr'][:])
    inv_cov_table_nubar = np.array(h5file_nubar['NubarMatr'][:])
    h5file_nu.close()
    h5file_nubar.close()

    inv_cov_matrix_nu = inv_cov_table_nu.reshape(
        map(len, (E_CENTERS, X_CENTERS, Y_CENTERS)) * 2
    )
    inv_cov_matrix_nubar = inv_cov_table_nubar.reshape(
        map(len, (E_CENTERS, X_CENTERS, Y_CENTERS)) * 2
    )

    return inv_cov_matrix_nu, inv_cov_matrix_nubar


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


# def evaluate(params_dict, a_sys, b_sys, nu=True):
def evaluate(params_dict, a_sys, nu=True):
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
    weights = np.power(params_dict['x_array'], -a_sys)# * b_sys

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
            sigma_nu += np.sum(unumpy.nominal_values(sigma_nu_x_e.data))

        if nu: scaling.append((WA_NU / 1E-38) / sigma_nu)
        else: scaling.append((WA_NUBAR / 1E-38) / sigma_nu)
    scaling = ma.masked_invalid(scaling)
    nu_histograms = ma.masked_equal(nu_histograms, 0)

    scaled_histograms = nu_histograms * \
            scaling[:, np.newaxis][:, np.newaxis]
    return scaled_histograms


def calculate_chi2(data, mc, corrmatr):
    data = unumpy.nominal_values(data).data.astype(float)
    mc = unumpy.nominal_values(mc).data.astype(float)
    bin_shape = map(len, (E_CENTERS, X_CENTERS, Y_CENTERS))
    diff = data - mc
    chi_squared = np.einsum('ijk,ijklmn,lmn->', diff, corrmatr, diff)

    return chi_squared


def wrap_calculation(systematics, args=None):
    NU_HISTOGRAMS = evaluate(
        # args['params_dict'], systematics[0], systematics[1], nu=args['nu']
        args['params_dict'], systematics, nu=args['nu']
    )
    chi_squared = calculate_chi2(
        args['data'], NU_HISTOGRAMS, args['inv_cov_matrix']
    )

    # if args['nu']: wa = WA_NU / 1E-38
    # else: wa = WA_NUBAR / 1E-38
    # x_bin_sizes = get_bin_sizes(X_BINNING).astype(float)
    # y_bin_sizes = get_bin_sizes(Y_BINNING).astype(float)
    # nuisance = 0
    # for e_idx in xrange(len(E_CENTERS)):
    #     integral = np.dot(np.dot(unumpy.nominal_values(NU_HISTOGRAMS[e_idx]),
    #                              y_bin_sizes), x_bin_sizes)
    #     nuisance += (((integral / wa) - 1) / 0.02) ** 2

    # chi_squared += nuisance
    logging.info('chi_squared = {0}\n'.format(chi_squared))
    return chi_squared


if __name__ == '__main__':
    set_verbosity(3)
    nu_data_matrix, nubar_data_matrix = load_nutev_xsec_vec()
    inv_cov_matrix_nu, inv_cov_matrix_nubar = load_nutev_corrmatrix()
    params_dict = load_genie_mc()
    params_dict = only_discc(params_dict)

    # minim_result = opt.minimize(
    #     fun=wrap_calculation,
    #     # x0=(0.0796178933896, 0.83061502),
    #     x0=0.0796178933896,
    #     args={'params_dict'    : params_dict,
    #           'data'           : nu_data_matrix,
    #           'inv_cov_matrix' : inv_cov_matrix_nu,
    #           'nu'             : True},
    #     method='L-BFGS-B',
    #     options={"disp"    : 0,
    #              "ftol"    : 2e-7,
    #              "eps"     : 1.0e-4,
    #              "gtol"    : 1.0e-5,
    #              "maxcor"  : 10,
    #              "maxfun"  : 15000,
    #              "maxiter" : 200}
    # )

    minim_result = opt.minimize(
        fun=wrap_calculation,
        # x0=(0.0796178933896, 0.83061502),
        x0=0.1,
        args={'params_dict'    : params_dict,
              'data'           : nubar_data_matrix,
              'inv_cov_matrix' : inv_cov_matrix_nubar,
              'nu'             : False},
        method='L-BFGS-B',
        options={"disp"    : 0,
                 "ftol"    : 2e-7,
                 "eps"     : 1.0e-4,
                 "gtol"    : 1.0e-5,
                 "maxcor"  : 10,
                 "maxfun"  : 15000,
                 "maxiter" : 200}
    )

    logging.info('{0}'.format(minim_result))
    # jacobian = np.array([1, minim_result.jac[0],
    #                      minim_result.jac[1], 1]).reshape((2, 2))
    # covariance = inv(np.dot(jacobian.T, jacobian))
    # logging.info('Covariance = {0}'.format(covariance))
