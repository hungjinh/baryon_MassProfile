import sys
sys.path.append("/Users/hhg/Research/c_lib/")
import illustris_python as il
import h5py
import numpy as np
from hydrosim_kit import *
import random

class TNG_info():
    def __init__(self, basePath, snap):
        self.basePath = basePath
        self.snap = snap
        self.header_snapCat = self.load_snap_header()
        self.header_groupCat = self.load_group_header()
        self.init_VIP_sim_info()

    def load_snap_header(self):
        fname_header = self.basePath + '/snapdir_%03d/snap_%03d.0.hdf5' %(self.snap, self.snap)
        with h5py.File(fname_header, 'r') as f:
            header_snapCat = dict(f['Header'].attrs.items())
        return header_snapCat

    def load_group_header(self):
        header_groupCat = il.groupcat.loadHeader(self.basePath, self.snap)
        return header_groupCat
    
    def init_VIP_sim_info(self):
        self.Lbox = self.header_snapCat['BoxSize']       # [kpc/h]
        self.redshift = self.header_snapCat['Redshift']
        self.isDMO = 1 if self.header_snapCat['NumPart_Total'][0] == 0 else 0

        self.Nptls = {}
        self.Nptls['dm'] = self.header_snapCat['NumPart_Total'][1]
        self.Nptls['gas'] = self.header_snapCat['NumPart_Total'][0]
        self.Nptls['star'] = self.header_snapCat['NumPart_Total'][4]
        self.Nptls['bh'] = self.header_snapCat['NumPart_Total'][5]
        self.Nptls['total'] = self.Nptls['dm']+self.Nptls['gas']+self.Nptls['star']+self.Nptls['bh']

        self.dm_ptl_mass = self.header_snapCat['MassTable'][1]   # [1e10 Msun/h]

        self.Nsubgroups = self.header_groupCat['Nsubgroups_Total']
        
        self.cosmology = {}
        self.cosmology['h'] = self.header_snapCat['HubbleParam']
        self.cosmology['Omega_b'] = self.header_snapCat['OmegaBaryon']
        self.cosmology['Omega_m'] = self.header_snapCat['Omega0']
        self.cosmology['Omega_l'] = self.header_snapCat['OmegaLambda']

        self.Nfiles_snap = self.header_snapCat['NumFilesPerSnapshot']
        self.Nfiles_group = self.header_groupCat['NumFiles']


class TNGsnap_info(TNG_info):
    def __init__(self, basePath, snap, subsample_frac=None):
        super().__init__(basePath, snap)

        self.subsample_frac = subsample_frac
        if self.subsample_frac is not None:
            self.takeout_ID = {}

        if self.isDMO==1 :
            self.snap_cat = {}
            self.snap_cat['pos'] = self._load_ptl_pos('dm')
        else :
            self.snap_cat = self.load_ptl_info_hydro()
        
    def random_sampling(self, Nptl):
        
        ID_ptl = list(range(Nptl))
        Nptl_sampled = int(Nptl*self.subsample_frac)
        takeout_ID = random.sample(ID_ptl, Nptl_sampled)

        return takeout_ID
    
    def _load_ptl_pos(self, partType):
        print('loading snapshot ptl:', partType)
        ptl_pos = il.snapshot.loadSubset(self.basePath, self.snap, partType, fields=['Coordinates'])

        if self.subsample_frac is not None:
            self.takeout_ID[partType] = self.random_sampling(self.Nptls[partType])
            ptl_pos = ptl_pos[self.takeout_ID[partType]]

        return ptl_pos
    
    def _load_ptl_mass(self, partType):

        if partType=='dm':
            return self.dm_ptl_mass
        
        else:
            ptl_mass = il.snapshot.loadSubset(self.basePath, self.snap, partType, fields=['Masses'])

            if self.subsample_frac is not None:
                ptl_mass = ptl_mass[self.takeout_ID[partType]]

            return ptl_mass
    
    def load_ptl_info_hydro(self):

        snap_cat = {}
        snap_cat['pos'] = self._load_ptl_pos('dm')
        snap_cat['mass'] = np.array([self.dm_ptl_mass]*len(snap_cat['pos']))
            
        for partType in ['gas', 'star', 'bh']:
            pos_baryon = self._load_ptl_pos(partType)
            snap_cat['pos'] = np.vstack((snap_cat['pos'], pos_baryon))
                
            mass_baryon = self._load_ptl_mass(partType)
            snap_cat['mass'] = np.hstack((snap_cat['mass'], mass_baryon))
                
        return snap_cat



class TNGgroup_info(TNG_info):

    def __init__(self, basePath, snap, log10_Mhalo_range=None):
        '''
            log10_Mhalo_range = [14.5, 16.]  # in unit of log10(Msun/h)
        '''
        super().__init__(basePath, snap)

        if log10_Mhalo_range is not None:
            self.Mhalo_range = self._unitTrans_mass(log10_Mhalo_range) # [1e10 Msun/h]
        else:
            self.Mhalo_range = None

        self.halo_cat = self.load_group_info(self.Mhalo_range)
    
    def _unitTrans_mass(self, log10_Msun):
        #transfrom the unit of mass input from [log10(Msun/h)] to [1e10 Msun/h]
        if len(log10_Msun)==1:
            return 10**log10_Msun/1e10
        else:
            return 10**np.array(log10_Msun)/1e10

    def load_group_info(self, Mhalo_range=None):
        '''
            column name info:
            ID_FoFhalo: the ID of the FoF halo catalog (when loaded it with il.groupcat.loadHalos, before taking subcatalog)
            GroupFirstSub: ID into the Subhalo table of the first/primary/most massive SUBFIND group within the FoF group
        '''

        fields = ['GroupFirstSub', 'GroupPos', 'Group_M_Mean200', 'Group_R_Mean200', 'GroupNsubs', 'GroupMass']
        halo_cat = il.groupcat.loadHalos(self.basePath, self.snap, fields=fields)
        halo_cat['ID_FoFhalo'] = np.arange(halo_cat['count'])
        del halo_cat['count']

        if Mhalo_range is not None:
            takeout_ID = np.where( (halo_cat['GroupMass'] >= Mhalo_range[0]) & (halo_cat['GroupMass'] < Mhalo_range[1]) ) 
            for colname in halo_cat.keys():
                halo_cat[colname] = halo_cat[colname][takeout_ID]
        
        if self.isDMO == 0:
            halo_cat['matched_ID_Subhalo_dmo'], halo_cat['matched_ID_FoFhalo_dmo'] = self.find_matched_HaloID_in_DMO(halo_cat['GroupFirstSub'])

        return halo_cat

    def find_matched_HaloID_in_DMO(self, GroupFirstSub_hydro):
        '''
            finding matched Subhalo ID into the DMO Subhalo table (il.groupcat.loadSubhalos), given hydro GroupFirstSub_hydro
            matched_ID_Subhalo_dmo: matched ID into the DMO Subhalo table
            matched_ID_FoFhalo_dmo: matched ID into the DMO FoF table 
            SubhaloGSubhaloGrNr_dmorNr_dmo: ID into the Group table of the FoF host/parent (ID_FoFhalo) of this DMO Subhalo
        '''

        self.ID_matched = load_subhalo_matching_to_dark(self.basePath, self.snap, key="SubhaloIndexDark_LHaloTree")
        matched_ID_Subhalo_dmo = self.ID_matched[GroupFirstSub_hydro]

        self.basePath_dmo = self.basePath + '-dark'
        SubhaloGrNr_dmo = il.groupcat.loadSubhalos(self.basePath_dmo, self.snap ,fields=['SubhaloGrNr']) # 'SubhaloParent','SubhaloCM','SubhaloMass'
        matched_ID_FoFhalo_dmo = SubhaloGrNr_dmo[matched_ID_Subhalo_dmo]

        return matched_ID_Subhalo_dmo, matched_ID_FoFhalo_dmo


class TNGmatched_groupCat():
    def __init__(self, basePath_hydro, basePath_dmo, snap, log10_Mhalo_range=None):
        self.GP_hydro = TNGgroup_info(basePath_hydro, snap, log10_Mhalo_range)
        self.GP_dmo = TNGgroup_info(basePath_dmo, snap, log10_Mhalo_range=None) 
                                    # need to load the whole DMO group catalog without Mass_range selection

        self.matched_halo_cat_hydro = self.kick_hydro_noMatched(halo_cat_hydro=self.GP_hydro.halo_cat)
        self.matched_halo_cat_dmo = self.select_dmo_Matched(halo_cat_dmo=self.GP_dmo.halo_cat)
    
    def kick_hydro_noMatched(self, halo_cat_hydro):
        '''
            kickout rows in halo_cat_hydro that do not have matched dmo subhalo
        '''
        halo_cat_hydro = halo_cat_hydro.copy()

        takeout_ID = np.where(halo_cat_hydro['matched_ID_Subhalo_dmo'] >= 0)

        for colname in halo_cat_hydro.keys():
                halo_cat_hydro[colname] = halo_cat_hydro[colname][takeout_ID]

        return halo_cat_hydro

    def select_dmo_Matched(self, halo_cat_dmo):
        '''
            creat a DMO halo catalog that matched with self.matched_halo_cat_hydro
        '''
        halo_cat_dmo = halo_cat_dmo.copy()
        
        takeout_ID = self.matched_halo_cat_hydro['matched_ID_FoFhalo_dmo']

        for colname in halo_cat_dmo.keys():
                halo_cat_dmo[colname] = halo_cat_dmo[colname][takeout_ID]

        return halo_cat_dmo



