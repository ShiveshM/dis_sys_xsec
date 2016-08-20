import sys, os

import h5py
import cPickle as pickle

if len(sys.argv) == 1:
    raise ValueError, "You need some arguments"
if len(sys.argv) == 2:
    raise ValueError, "You need more arguments"
if len(sys.argv) > 3:
    raise ValueError, "You have too many arguments"

infile = sys.argv[1]
outfile = sys.argv[2]

pkl = pickle.load(open(infile, 'rb'))

h5 = h5py.File(outfile, 'w')
for key in pkl.iterkeys():
    try:
        h5.create_dataset(key, data=pkl[key])
    except:
        print key, pkl[key]
        raise ValueError
h5.close()
