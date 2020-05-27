import sys
sys.path.append("/Users/hhg/Research/c_lib/")
import illustris_python as il
import treecorr
from hydrosim_kit import *
from load_catalog import *
import time

class Xi_treecorr():
    def __init__(self, halo_cat, snap_cat, Lbox):
        '''
            halo_cat : a dict contains the halo info ['GroupPos', 'GroupMass']
            snap_cat : a dict contains snapshot particle info ['pos', 'mass']
        '''

        self.Lbox = Lbox
        self.Nptl = len(snap_cat['pos'])
        self.halo_cat = halo_cat
        self.snap_cat = snap_cat

        self.isDMO = 0 if 'mass' in snap_cat.keys() else 1

        self.Nptl_random = self.Nptl

        self.rmin = 10. #[kpc/h]
        self.rmax = 3000.


    def gen_Catalog_random(self, N):

        pos_random = np.random.uniform(0, self.Lbox, size=(N, 3))
        R = treecorr.Catalog(x=pos_random[:, 0], y=pos_random[:, 1], z=pos_random[:, 2])

        return R
    
    def gen_Catalog_Dhalo(self):
        
        pos_halo = self.halo_cat['GroupPos']
        wt_halo = self.halo_cat['GroupMass'].copy()
        wt_halo /= wt_halo.max()

        D1 = treecorr.Catalog(x=pos_halo[:, 0], y=pos_halo[:, 1], z=pos_halo[:, 2], w=wt_halo)

        return D1
    
    def gen_Catalog_Dptl(self):

        pos_ptl = self.snap_cat['pos']

        if self.isDMO == 0 :
            wt_ptl = self.snap_cat['mass'].copy()
            wt_ptl/=wt_ptl.max()
            
            D2 = treecorr.Catalog(x=pos_ptl[:, 0], y=pos_ptl[:, 1], z=pos_ptl[:, 2], w=wt_ptl)
        else:
            D2 = treecorr.Catalog(x=pos_ptl[:, 0], y=pos_ptl[:, 1], z=pos_ptl[:, 2])
        return D2
        


    def run_treecorr(self):

        R = self.gen_Catalog_random(N=self.Nptl_random)
        D1 = self.gen_Catalog_Dhalo()
        D2 = self.gen_Catalog_Dptl()

        rmin = self.rmin
        rmax = self.rmax
        nbins = 10
        metric = 'Euclidean'

        # match the units of r with units of position
        self.DD = treecorr.NNCorrelation(min_sep=rmin, max_sep=rmax, nbins=nbins)
        self.DR = treecorr.NNCorrelation(min_sep=rmin, max_sep=rmax, nbins=nbins)
        self.RR = treecorr.NNCorrelation(min_sep=rmin, max_sep=rmax, nbins=nbins)
        self.RD = treecorr.NNCorrelation(min_sep=rmin, max_sep=rmax, nbins=nbins)

        self.DD.process(D1, D2, metric=metric)
        self.DR.process(D1, R, metric=metric)
        self.RR.process(R, R, metric=metric)
        self.RD.process(R, D2, metric=metric)

        xi, varxi = self.DD.calculateXi(self.RR, self.DR, self.RD)

        return self.DD.meanr, xi, varxi








