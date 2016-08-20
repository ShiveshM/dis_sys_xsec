import sys, os

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

if len(sys.argv) == 1:
    raise ValueError, "You need some arguments"
infile = sys.argv[1]

def make_plot(hist, tot_hist, out_name):
    print 'hist', hist
    print 'hist', hist.shape
    print 'tot_hist', tot_hist
    print 'tot_hist', tot_hist.shape
    percent_dis = hist / tot_hist.astype(float)
    print 'percent_dis', percent_dis
    print 'percent_dis', percent_dis.shape

    fig = plt.figure(figsize=[12, 8]) 
    ax  = fig.add_subplot(111)
    fig.suptitle('/data/ana/LE/NBI_nutau_appearance/level7_24Nov2015/'+out_name+'/', y=1.005)
    ax.grid(True)
    plt.xlabel(r'True E$_{\nu}$(GeV)', size=18)
    plt.ylabel(r'% DIS (CC)', size=18)
    ax.set_xlim([1E0, 1E3])
    ax.set_ylim([0, 1.1])
    ax.set_xscale('log')

    zero_numpy_array_element = np.array([0])
    hist = np.hstack((percent_dis, zero_numpy_array_element))                                                                                                                           
    ax.step(binning, hist, where='post')

    fig.tight_layout()
    fig.savefig('./test.pdf', bbox_inches='tight')
    # fig.savefig('./'+out_name+'.png', bbox_inches='tight')

def load_file(filename):
    h5file = h5py.File(filename, 'rb')
    x = {}
    for key in h5file.iterkeys():
        x[key] = h5file[key][:]
    logging.info('Loaded {0} events'.format(len(x['energy'])))
    return x

params = load_file(infile)
print params
print params['dis'].shape

binning = np.logspace(0, 3, 30)

cc_map = params['cc'].astype(bool)
dis_hist, edges = np.histogram(params['energy'][cc_map], bins=binning, weights=params['dis'][cc_map].astype(int))
all_hist, edges = np.histogram(params['energy'][cc_map], bins=binning)

# make_plot(dis_hist, all_hist, '12585')
make_plot(dis_hist, all_hist, '14585')
# make_plot(dis_hist, all_hist, '16585')
