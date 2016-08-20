
import sys, os
from copy import deepcopy

import numpy as np
import h5py

import icecube
from icecube import dataio, dataclasses, icetray
# from icecube import genie_icetray

from DXSec.flux import Flux
atmo_flux = Flux.service_factory(
    'honda', flux_file='/data/icecube/data/honda_flux/spl-ally-20-12-solmax.d'
)

if len(sys.argv) == 1:
    raise ValueError, "You need some arguments"

outfile = sys.argv[-1]
indir = sys.argv[1]


def load_file(filepath, **kwargs):
    print 'Loading I3 file {0} '.format(filepath)

    parameters = {}
    # parameters['energy'] = []
    parameters['weights'] = []

    inFile = dataio.I3File(filepath, 'r')

    while(inFile.more()):
        # frame = inFile.pop_physics()
        frame = inFile.pop_daq()
        if str(frame) == 'None':
            inFile.rewind()
            break
	
        def fix_type(s):
            if 'Bar' in s:
                return s.lower().replace('bar', '_bar')
            return s.lower()
        mctree = frame['I3MCTree']
        neutrino = dataclasses.get_most_energetic_neutrino(mctree)
        nu_type = fix_type(str(neutrino.type))
        nu_energy = neutrino.energy
        nu_czenith = np.cos(neutrino.dir.zenith)
        nu_azimuth = neutrino.dir.azimuth * 180 / np.pi

        flux_value = atmo_flux.get_flux(
            nu_type, nu_energy, nu_czenith, nu_azimuth
        ) / pow(100, 2)

        oneweight = frame['I3MCWeightDict']['OneWeight']
        nevents = frame['I3MCWeightDict']['NEvents']

        if 'bar' not in nu_type:
            onew_pertype = oneweight / 0.7
        else:
            onew_pertype = oneweight / 0.3
        parameters['weights'].append(
            (onew_pertype * flux_value) / nevents
        )

	# nu = frame['MCNeutrino']

	# parameters['energy'].append(nu.energy)
        # for parm in frame['I3GENIEResultDict'].iterkeys():
            # try:
                # parameters[parm].append(frame['I3GENIEResultDict'][parm])
            # except:
                # parameters[parm] = []
                # parameters[parm].append(frame['I3GENIEResultDict'][parm])
    inFile.close()

    for key in parameters.iterkeys():
        parameters[key] = np.array(parameters[key])
    print 'Loaded {0} events'.format(len(parameters['weights']))
    # print 'Loaded {0} events'.format(len(parameters['dis']))

    return parameters

def load_directory(dirpath, **kwargs):
    """
    Load data from all I3 files inside a given directory
    """
    print 'Loading I3 files from directory \n' \
                 '           {0}'.format(dirpath)
    parameters = {}

    def merge(x, y):
        return np.concatenate((x, y))

    n_files = 0
    for file_name in sorted(os.listdir(dirpath)):
        file_path = dirpath + '/' + file_name
        if os.path.isfile(file_path) \
           and (file_path.endswith('.i3.bz2') or
                file_path.endswith('.i3.gz')):
            n_files += 1
	    # if n_files == 2: break
            sub_parameters = load_file(file_path, **kwargs)
            for key in sub_parameters.iterkeys():
                try:
                    parameters[key] = merge(parameters[key],
                                            sub_parameters[key])
                except:
                    parameters[key] = deepcopy(sub_parameters[key])
    print 'Loaded {0} events in total'.format(len(parameters['weights']))
    # print parameters.keys()
    # print 'Loaded {0} events in ' \
    #              'total'.format(len(parameters['dis']))

    return parameters

def cache_parameters(parameters, filepath):
    """
    Store the parameters dictionary to a HDF5 file
    """
    print 'Caching parameters to file {0}'.format(filepath)
    h5file = h5py.File(filepath, 'w')

    for key in parameters.iterkeys():
        h5file.create_dataset(key, data=parameters[key])
    h5file.close()

arr = load_directory(indir)
cache_parameters(arr, outfile)
