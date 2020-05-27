import sys
sys.path.append("/Users/hhg/Research/c_lib/illustris_python/")
import illustris_python as il
import h5py
import numpy as np
import random

def subsample_snapshot(index_arr, frac):
    '''
        subsampling a fraction of particles.
        index_arr: an indel array to be subsampling
        frac: fraction of particles to be takeout
    '''
    Nptls = len(index_arr)
    Nout = int(frac*Nptls)
    return random.sample(index_arr, Nout)

def load_subhalo_matching_to_dark(basePath, snap, key="SubhaloIndexDark_LHaloTree"):
    '''
        basePath needs to be a hydro directory

        Data format: 
            len(Nrows): equal to the number of subhalos at the corresponding snapshot
            columns : provide the results of two different matching algorithms
                      Each array gives integer indices, whose value is the corresponding index of the matched subhalo 
        
            key = 'SubhaloIndexDark_LHaloTree' -- based on the LHaloTree matching algorithm
                The matching is bidirectional, i.e. TNG <-> DMO. In each base, the best subhalo candidate is chosen as that with the largest number of matching DM particles (α=0). Only if the candidate in each direction agrees (bijective), then these matches are saved.
            
            key = 'SubhaloIndexDark_SubLink' -- based on the SubLink weighting algorithm
                The direction of the matching is TNG -> DMO, i.e. for each subhalo in the baryonic physics box a best match is found in the DMO run.

        (check Lovell+18)
    '''
    fname = basePath+"/subhalo_matching_to_dark.hdf5"

    hf = h5py.File(fname, 'r')
    print("available keys:", hf.get(u'Snapshot_%02d'%snap).keys())
    ID_matched = np.array(hf[u'Snapshot_%02d'%snap][key])

    return ID_matched
