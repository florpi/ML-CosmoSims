# code for reading Subfind's subhalo_tab files
# usage e.g.:
#
# import readsubf
# cat = readsubf.subfind_catalog("./m_10002_h_94_501_z3_csf/",63,masstab=True)
# print( cat.nsubs
# print( "largest halo x position = ",cat.sub_pos[0][0] 

import numpy as np
import os
import sys
 
class subfind_catalog:
  def __init__(self, basedir, snapnum, group_veldisp = False, masstab = False, long_ids = False, swap = False): # all optional parameters are not needed for HDF5 catalogues
    self.swap = swap
 
    self.filebase = basedir + "/groups_" + str(snapnum).zfill(3) + "/subhalo_tab_" + str(snapnum).zfill(3) + "."
    self.idbase = basedir + "/groups_" + str(snapnum).zfill(3) + "/subhalo_ids_" + str(snapnum).zfill(3) + "."
 
    curfile = self.filebase + str("0")  
    if (not os.path.exists(curfile)):
      if os.path.exists(basedir + "/fof_subhalo_tab_" + str(snapnum).zfill(3) + ".hdf5"):
        self.is_hdf5 = True
        self.filebase = basedir + "/fof_subhalo_tab_" + str(snapnum).zfill(3) + "."
      elif os.path.exists(basedir + "/groups_" + str(snapnum).zfill(3) + "/fof_subhalo_tab_" + str(snapnum).zfill(3) + ".0.hdf5"):
        self.is_hdf5 = True
        self.filebase = basedir + "/groups_" + str(snapnum).zfill(3) + "/fof_subhalo_tab_" + str(snapnum).zfill(3) + "."
      else:
        print( "file not found:", curfile, basedir + "/fof_subhalo_tab_" + str(snapnum).zfill(3) + ".hdf5",basedir + "/groups_" + str(snapnum).zfill(3) + "/fof_subhalo_tab_" + str(snapnum).zfill(3) + ".0.hdf5")
        sys.exit()
    else:
      self.is_hdf5 = False

    print()
    print( "reading subfind catalog for snapshot",snapnum,"of",basedir)

    if (not self.is_hdf5): # standard file
      
      if long_ids: self.id_type = np.uint64
      else: self.id_type = np.uint32
  
      self.group_veldisp = group_veldisp
      self.masstab = masstab
  
      filenum = 0
      doneflag = False
      skip_gr = 0
      skip_sub = 0
      while not doneflag:
        curfile = self.filebase + str(filenum)
        
        if (not os.path.exists(curfile)):
          print( "file not found:", curfile)
          sys.exit()
        
        f = open(curfile,'rb')
                
        ngroups = np.fromfile(f, dtype=np.uint32, count=1)[0]
        totngroups = np.fromfile(f, dtype=np.uint32, count=1)[0]
        nids = np.fromfile(f, dtype=np.uint32, count=1)[0]
        totnids = np.fromfile(f, dtype=np.uint64, count=1)[0]
        ntask = np.fromfile(f, dtype=np.uint32, count=1)[0]
        nsubs = np.fromfile(f, dtype=np.uint32, count=1)[0]
        totnsubs = np.fromfile(f, dtype=np.uint32, count=1)[0]
        
        if swap:
          ngroups = ngroups.byteswap()
          totngroups = totngroups.byteswap()
          nids = nids.byteswap()
          totnids = totnids.byteswap()
          ntask = ntask.byteswap()
          nsubs = nsubs.byteswap()
          totnsubs = totnsubs.byteswap()
        
        if filenum == 0:
          self.ngroups = totngroups
          self.nids = totnids
          self.nfiles = ntask
          self.nsubs = totnsubs

          self.group_len = np.empty(totngroups, dtype=np.uint32)
          self.group_offset = np.empty(totngroups, dtype=np.uint32)
          self.group_mass = np.empty(totngroups, dtype=np.float64)
          self.group_pos = np.empty(totngroups, dtype=np.dtype((np.float64,3)))
          self.group_m_mean200 = np.empty(totngroups, dtype=np.float64)
          self.group_r_mean200 = np.empty(totngroups, dtype=np.float64)
          self.group_m_crit200 = np.empty(totngroups, dtype=np.float64)
          self.group_r_crit200 = np.empty(totngroups, dtype=np.float64)
          self.group_m_tophat200 = np.empty(totngroups, dtype=np.float64)
          self.group_r_tophat200 = np.empty(totngroups, dtype=np.float64)
          if group_veldisp:
            self.group_veldisp_mean200 = np.empty(totngroups, dtype=np.float64)
            self.group_veldisp_crit200 = np.empty(totngroups, dtype=np.float64)
            self.group_veldisp_tophat200 = np.empty(totngroups, dtype=np.float64)
          self.group_contamination_count = np.empty(totngroups, dtype=np.uint32)
          self.group_contamination_mass = np.empty(totngroups, dtype=np.float64)
          self.group_nsubs = np.empty(totngroups, dtype=np.uint32)
          self.group_firstsub = np.empty(totngroups, dtype=np.uint32)
          
          self.sub_len = np.empty(totnsubs, dtype=np.uint32)
          self.sub_offset = np.empty(totnsubs, dtype=np.uint32)
          self.sub_parent = np.empty(totnsubs, dtype=np.uint32)
          self.sub_mass = np.empty(totnsubs, dtype=np.float64)
          self.sub_pos = np.empty(totnsubs, dtype=np.dtype((np.float64,3)))
          self.sub_vel = np.empty(totnsubs, dtype=np.dtype((np.float64,3)))
          self.sub_cm = np.empty(totnsubs, dtype=np.dtype((np.float64,3)))
          self.sub_spin = np.empty(totnsubs, dtype=np.dtype((np.float64,3)))
          self.sub_veldisp = np.empty(totnsubs, dtype=np.float64)
          self.sub_vmax = np.empty(totnsubs, dtype=np.float64)
          self.sub_vmaxrad = np.empty(totnsubs, dtype=np.float64)
          self.sub_halfmassrad = np.empty(totnsubs, dtype=np.float64)
          self.sub_id_mostbound = np.empty(totnsubs, dtype=self.id_type)
          self.sub_grnr = np.empty(totnsubs, dtype=np.uint32)
          if masstab:
            self.sub_masstab = np.empty(totnsubs, dtype=np.dtype((np.float64,6)))
      
        if ngroups > 0:
          locs = slice(skip_gr, skip_gr + ngroups)
          self.group_len[locs] = np.fromfile(f, dtype=np.uint32, count=ngroups)
          self.group_offset[locs] = np.fromfile(f, dtype=np.uint32, count=ngroups)
          self.group_mass[locs] = np.fromfile(f, dtype=np.float64, count=ngroups)
          self.group_pos[locs] = np.fromfile(f, dtype=np.dtype((np.float64,3)), count=ngroups)
          self.group_m_mean200[locs] = np.fromfile(f, dtype=np.float64, count=ngroups)
          self.group_r_mean200[locs] = np.fromfile(f, dtype=np.float64, count=ngroups)
          self.group_m_crit200[locs] = np.fromfile(f, dtype=np.float64, count=ngroups)
          self.group_r_crit200[locs] = np.fromfile(f, dtype=np.float64, count=ngroups)
          self.group_m_tophat200[locs] = np.fromfile(f, dtype=np.float64, count=ngroups)
          self.group_r_tophat200[locs] = np.fromfile(f, dtype=np.float64, count=ngroups)
          if group_veldisp:
            self.group_veldisp_mean200[locs] = np.fromfile(f, dtype=np.float64, count=ngroups)
            self.group_veldisp_crit200[locs] = np.fromfile(f, dtype=np.float64, count=ngroups)
            self.group_veldisp_tophat200[locs] = np.fromfile(f, dtype=np.float64, count=ngroups)
          self.group_contamination_count[locs] = np.fromfile(f, dtype=np.uint32, count=ngroups)
          self.group_contamination_mass[locs] = np.fromfile(f, dtype=np.float64, count=ngroups)
          self.group_nsubs[locs] = np.fromfile(f, dtype=np.uint32, count=ngroups)
          self.group_firstsub[locs] = np.fromfile(f, dtype=np.uint32, count=ngroups)        
          skip_gr += ngroups
          
        if nsubs > 0:
          locs = slice(skip_sub, skip_sub + nsubs)
          self.sub_len[locs] = np.fromfile(f, dtype=np.uint32, count=nsubs)
          self.sub_offset[locs] = np.fromfile(f, dtype=np.uint32, count=nsubs)
          self.sub_parent[locs] = np.fromfile(f, dtype=np.uint32, count=nsubs)
          self.sub_mass[locs] = np.fromfile(f, dtype=np.float64, count=nsubs)
          self.sub_pos[locs] = np.fromfile(f, dtype=np.dtype((np.float64,3)), count=nsubs)
          self.sub_vel[locs] = np.fromfile(f, dtype=np.dtype((np.float64,3)), count=nsubs)
          self.sub_cm[locs] = np.fromfile(f, dtype=np.dtype((np.float64,3)), count=nsubs)
          self.sub_spin[locs] = np.fromfile(f, dtype=np.dtype((np.float64,3)), count=nsubs)
          self.sub_veldisp[locs] = np.fromfile(f, dtype=np.float64, count=nsubs)
          self.sub_vmax[locs] = np.fromfile(f, dtype=np.float64, count=nsubs)
          self.sub_vmaxrad[locs] = np.fromfile(f, dtype=np.float64, count=nsubs)
          self.sub_halfmassrad[locs] = np.fromfile(f, dtype=np.float64, count=nsubs)
          self.sub_id_mostbound[locs] = np.fromfile(f, dtype=self.id_type, count=nsubs)
          self.sub_grnr[locs] = np.fromfile(f, dtype=np.uint32, count=nsubs)
          if masstab:
            self.sub_masstab[locs] = np.fromfile(f, dtype=np.dtype((np.float64,6)), count=nsubs)
          skip_sub += nsubs

        curpos = f.tell()
        f.seek(0,os.SEEK_END)
        if curpos != f.tell(): print( "Warning: finished reading before EOF for file",filenum)
        f.close()  
        #print( 'finished with file number',filenum,"of",ntask
        filenum += 1
        if filenum == self.nfiles:
          doneflag = True
        
      if swap:
        self.group_len.byteswap(True)
        self.group_offset.byteswap(True)
        self.group_mass.byteswap(True)
        self.group_pos.byteswap(True)
        self.group_m_mean200.byteswap(True)
        self.group_r_mean200.byteswap(True)
        self.group_m_crit200.byteswap(True)
        self.group_r_crit200.byteswap(True)
        self.group_m_tophat200.byteswap(True)
        self.group_r_tophat200.byteswap(True)
        if group_veldisp:
          self.group_veldisp_mean200.byteswap(True)
          self.group_veldisp_crit200.byteswap(True)
          self.group_veldisp_tophat200.byteswap(True)
        self.group_contamination_count.byteswap(True)
        self.group_contamination_mass.byteswap(True)
        self.group_nsubs.byteswap(True)
        self.group_firstsub.byteswap(True)
          
        self.sub_len.byteswap(True)
        self.sub_offset.byteswap(True)
        self.sub_parent.byteswap(True)
        self.sub_mass.byteswap(True)
        self.sub_pos.byteswap(True)
        self.sub_vel.byteswap(True)
        self.sub_cm.byteswap(True)
        self.sub_spin.byteswap(True)
        self.sub_veldisp.byteswap(True)
        self.sub_vmax.byteswap(True)
        self.sub_vmaxrad.byteswap(True)
        self.sub_halfmassrad.byteswap(True)
        self.sub_id_mostbound.byteswap(True)
        self.sub_grnr.byteswap(True)
        if masstab:
          self.sub_masstab.byteswap(True)
        
      print()
      print( "number of groups =", self.ngroups)
      print( "number of subgroups =", self.nsubs)
      if self.nsubs > 0:
        print( "largest group of length",self.group_len[0],"has",self.group_nsubs[0],"subhalos")
        print()
    
    else: # HDF5 file
      
      import h5py
      
      self.masstab = True
      
      filenum = 0
      doneflag = False
      skip_gr = 0
      skip_sub = 0
      
      while not doneflag:
        curfile = self.filebase + str(filenum) + ".hdf5"
        
        if filenum == 0:
          if not os.path.exists(curfile):
            curfile = self.filebase + "hdf5"

        f = h5py.File(curfile, "r")
        
        totngroups = f["Header"].attrs["Ngroups_Total"]
        totnsubs = f["Header"].attrs["Nsubgroups_Total"]
        ngroups = f["Header"].attrs["Ngroups_ThisFile"]
        nsubs = f["Header"].attrs["Nsubgroups_ThisFile"]
        
        if filenum == 0:
          self.ngroups = f["Header"].attrs["Ngroups_Total"]
          self.nids = f["Header"].attrs["Nids_Total"]
          self.nfiles = f["Header"].attrs["NumFiles"]
          self.nsubs = f["Header"].attrs["Nsubgroups_Total"]

          self.id_type = f["Subhalo/SubhaloIDMostbound"].value.dtype

          self.group_len = np.empty(totngroups, dtype=np.uint32)
          self.group_lentab = np.empty(totngroups, dtype=((np.uint32,6)))
          self.group_mass = np.empty(totngroups, dtype=np.float64)
          self.group_masstab = np.empty(totngroups, dtype=((np.float64,6)))
          self.group_pos = np.empty(totngroups, dtype=np.dtype((np.float64,3)))
          self.group_vel = np.empty(totngroups, dtype=np.dtype((np.float64,3)))
          self.group_m_mean200 = np.empty(totngroups, dtype=np.float64)
          self.group_r_mean200 = np.empty(totngroups, dtype=np.float64)
          self.group_m_crit200 = np.empty(totngroups, dtype=np.float64)
          self.group_r_crit200 = np.empty(totngroups, dtype=np.float64)
          self.group_m_crit500 = np.empty(totngroups, dtype=np.float64)
          self.group_r_crit500 = np.empty(totngroups, dtype=np.float64)
          self.group_m_tophat200 = np.empty(totngroups, dtype=np.float64)
          self.group_r_tophat200 = np.empty(totngroups, dtype=np.float64)
          self.group_nsubs = np.empty(totngroups, dtype=np.uint32)
          self.group_firstsub = np.empty(totngroups, dtype=np.uint32)
          #self.group_sfr = np.empty(totngroups, dtype=np.float64)
          
          self.sub_len = np.empty(totnsubs, dtype=np.uint32)
          self.sub_lentab = np.empty(totnsubs, dtype=((np.uint32,6)))
          self.sub_parent = np.empty(totnsubs, dtype=np.uint32)
          self.sub_mass = np.empty(totnsubs, dtype=np.float64)
          self.sub_masstab = np.empty(totnsubs, dtype=((np.float64,6)))
          self.sub_pos = np.empty(totnsubs, dtype=np.dtype((np.float64,3)))
          self.sub_vel = np.empty(totnsubs, dtype=np.dtype((np.float64,3)))
          self.sub_cm = np.empty(totnsubs, dtype=np.dtype((np.float64,3)))
          self.sub_spin = np.empty(totnsubs, dtype=np.dtype((np.float64,3)))
          self.sub_veldisp = np.empty(totnsubs, dtype=np.float64)
          self.sub_vmax = np.empty(totnsubs, dtype=np.float64)
          self.sub_vmaxrad = np.empty(totnsubs, dtype=np.float64)
          self.sub_halfmassrad = np.empty(totnsubs, dtype=np.float64)
          self.sub_id_mostbound = np.empty(totnsubs, dtype=self.id_type)
          self.sub_grnr = np.empty(totnsubs, dtype=np.uint32)
      
        if ngroups > 0:
          locs = slice(skip_gr, skip_gr + ngroups)
          self.group_len[locs] = f["Group/GroupLen"].value
          self.group_lentab[locs] = f["Group/GroupLenType"].value
          self.group_mass[locs] = f["Group/GroupMass"].value
          self.group_masstab[locs] = f["Group/GroupMassType"].value
          self.group_pos[locs] = f["Group/GroupPos"].value
          self.group_vel[locs] = f["Group/GroupVel"].value
          self.group_m_mean200[locs] = f["Group/Group_M_Mean200"].value
          self.group_r_mean200[locs] = f["Group/Group_R_Mean200"].value
          self.group_m_crit200[locs] = f["Group/Group_M_Crit200"].value
          self.group_r_crit200[locs] = f["Group/Group_R_Crit200"].value
          self.group_m_crit500[locs] = f["Group/Group_M_Crit500"].value
          self.group_r_crit500[locs] = f["Group/Group_R_Crit500"].value
          self.group_m_tophat200[locs] = f["Group/Group_M_TopHat200"].value
          self.group_r_tophat200[locs] = f["Group/Group_R_TopHat200"].value
          self.group_nsubs[locs] = f["Group/GroupNsubs"].value
          self.group_firstsub[locs] = f["Group/GroupFirstSub"].value
          #self.group_sfr = f["Group/GroupSFR"].value
          skip_gr += ngroups
          
        if nsubs > 0:
          locs = slice(skip_sub, skip_sub + nsubs)
          self.sub_len[locs] = f["Subhalo/SubhaloLen"].value
          self.sub_lentab[locs] = f["Subhalo/SubhaloLenType"].value
          self.sub_parent[locs] = f["Subhalo/SubhaloParent"].value
          self.sub_mass[locs] = f["Subhalo/SubhaloMass"].value
          self.sub_masstab[locs] = f["Subhalo/SubhaloMassType"].value
          self.sub_pos[locs] = f["Subhalo/SubhaloPos"].value
          self.sub_vel[locs] = f["Subhalo/SubhaloVel"].value
          self.sub_cm[locs] = f["Subhalo/SubhaloCM"].value
          self.sub_spin[locs] = f["Subhalo/SubhaloSpin"].value
          self.sub_veldisp[locs] = f["Subhalo/SubhaloVelDisp"].value
          self.sub_vmax[locs] = f["Subhalo/SubhaloVmax"].value
          self.sub_vmaxrad[locs] = f["Subhalo/SubhaloVmaxRad"].value
          self.sub_halfmassrad[locs] = f["Subhalo/SubhaloHalfmassRad"].value
          self.sub_id_mostbound[locs] = f["Subhalo/SubhaloIDMostbound"].value
          self.sub_grnr[locs] = f["Subhalo/SubhaloGrNr"].value
          skip_sub += nsubs
        
        f.close()  
        filenum += 1
        if filenum == self.nfiles: 
          doneflag = True
      
      self.group_offsettab = np.append(np.array([[0,0,0,0,0,0]],dtype=np.uint32),self.group_lentab.cumsum(axis=0)[0:-1],axis=0) 
      self.sub_offsettab = np.append(np.array([[0,0,0,0,0,0]],dtype=np.uint32),self.sub_lentab.cumsum(axis=0)[0:-1],axis=0)            
      for curgr in np.arange(self.ngroups):
        if self.group_nsubs[curgr] >= 1:
          cur_add_offsettab = self.group_offsettab[curgr] - self.sub_offsettab[self.group_firstsub[curgr]]  
          self.sub_offsettab[self.group_firstsub[curgr]:self.group_firstsub[curgr]+self.group_nsubs[curgr]] += cur_add_offsettab
      
    print()
    print( "number of groups =", self.ngroups)
    print( "number of subgroups =", self.nsubs)
    if self.nsubs > 0:
      print( "largest group of length",self.group_len[0],"has",self.group_nsubs[0],"subhalos")
      print()
        
  def read_ids(self):
    self.ids = np.empty(self.nids,dtype=self.id_type)
    
    if self.nids>0:    
      filenum = 0
      doneflag = False
      skip_ids = 0
      while not doneflag:
        curfile = self.idbase + str(filenum)
        
        if (not os.path.exists(curfile)):
          print( "file not found:", curfile)
          sys.exit()
        
        f = open(curfile,'rb')

        idngroups = np.fromfile(f, dtype=np.uint32, count=1)[0]
        idtotngroups = np.fromfile(f, dtype=np.uint32, count=1)[0]
        idnids = np.fromfile(f, dtype=np.uint32, count=1)[0]
        idtotnids = np.fromfile(f, dtype=np.uint64, count=1)[0]
        idntask = np.fromfile(f, dtype=np.uint32, count=1)[0]
        idoffset = np.fromfile(f, dtype=np.uint32, count=1)[0]

        if self.swap:
          idngroups = idngroups.byteswap()
          idtotngroups = idtotngroups.byteswap()
          idnids = idnids.byteswap()
          idtotnids = idtotnids.byteswap()
          idntask = idntask.byteswap()
          idoffset = idoffset.byteswap()

        assert skip_ids == idoffset
        
        self.ids[skip_ids:skip_ids+idnids] = np.fromfile(f, dtype=self.id_type, count=idnids)
        skip_ids += idnids

        filenum += 1
        if filenum == self.nfiles: doneflag = True
      assert skip_ids == self.nids
      if self.swap:
        self.ids.byteswap(True)
    else:
      print( "there are no IDs in this SUBFIND output!")
