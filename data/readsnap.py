"""
# routines for reading headers and data blocks from Gadget snapshot files
# usage e.g.:
#
# import readsnap as rs
# header = rs.snapshot_header("snap_063.0") # reads snapshot header
# print( header.massarr
# mass = rs.read_block("snap_063","MASS",parttype=5) # reads mass for particles of type 5, using block names should work for both format 1 and 2 snapshots
# print( "mass for", mass.size, "particles read"
# print( mass[0:10]
#
# before using read_block, make sure that the description (and order if using format 1 snapshot files) of the data blocks
# is correct for your configuration of Gadget 
#
# for mutliple file snapshots give e.g. the filename "snap_063" rather than "snap_063.0" to read_block
# for snapshot_header the file number should be included, e.g."snap_063.0", as the headers of the files differ
#
# the returned data block is ordered by particle species even when read from a multiple file snapshot

Important Informations:

Partype -1 = All PartTypes
Partype 0 = Gas
Partype 1 = Dark Matter
Partype 3 = Tracer -> does not have mass
Partype 4 = Stars & Wind (pos. & neg. age respectively)
Partype 5 = Black Holes

"""
import numpy as np
import os
import sys
import math
  
# ----- class for snapshot header ----- 

class snapshot_header:
  def __init__(self, filename, cosmic_ray_species=0):
    if (not os.path.exists(filename)):
      orig_filename = filename
      if os.path.exists(filename+".hdf5"):
          filename = filename+".hdf5"
      elif os.path.exists(filename+".0.hdf5"):
          filename = filename+".0.hdf5"
      
      if (not os.path.exists(filename)):
        print( "snapshot_header: files not found -> ", orig_filename, "or", filename)
        sys.exit()
    
    self.filename = filename
    if filename[-4:]!="hdf5":
      f = open(filename,'rb')    
      blocksize = np.fromfile(f,dtype=np.int32,count=1)
      if blocksize[0] == 8:
        swap = 0
        format = 2
      elif blocksize[0] == 256:
        swap = 0
        format = 1  
      else:
        blocksize.byteswap(True)
        if blocksize[0] == 8:
          swap = 1
          format = 2
        elif blocksize[0] == 256:
          swap = 1
          format = 1
        else:
            print("incorrect file format encountered when reading header of", filename)
            sys.exit()
      
      self.format = format
      self.swap = swap
      
      if format==2:
        f.seek(16, os.SEEK_CUR)
      
      self.npart = np.fromfile(f,dtype=np.int32,count=6)
      self.massarr = np.fromfile(f,dtype=np.float64,count=6)
      if cosmic_ray_species>0:
        self.cr_spectral_indices = np.fromfile(f,dtype=np.float64,count=cosmic_ray_species)
      self.time = (np.fromfile(f,dtype=np.float64,count=1))[0]
      self.redshift = (np.fromfile(f,dtype=np.float64,count=1))[0]
      self.sfr = (np.fromfile(f,dtype=np.int32,count=1))[0]
      self.feedback = (np.fromfile(f,dtype=np.int32,count=1))[0]
      self.nall = np.fromfile(f,dtype=np.int32,count=6)
      self.cooling = (np.fromfile(f,dtype=np.int32,count=1))[0]
      self.filenum = (np.fromfile(f,dtype=np.int32,count=1))[0]
      self.boxsize = (np.fromfile(f,dtype=np.float64,count=1))[0]
      self.omega_m = (np.fromfile(f,dtype=np.float64,count=1))[0]
      self.omega_l = (np.fromfile(f,dtype=np.float64,count=1))[0]
      self.hubble = (np.fromfile(f,dtype=np.float64,count=1))[0]
      
      if swap:
        self.npart.byteswap(True)
        self.massarr.byteswap(True)
        if cosmic_ray_species>0:
          self.cr_spectral_indices.byteswap(True)
        self.time = self.time.byteswap()
        self.redshift = self.redshift.byteswap()
        self.sfr = self.sfr.byteswap()
        self.feedback = self.feedback.byteswap()
        self.nall.byteswap(True)
        self.cooling = self.cooling.byteswap()
        self.filenum = self.filenum.byteswap()
        self.boxsize = self.boxsize.byteswap()
        self.omega_m = self.omega_m.byteswap()
        self.omega_l = self.omega_l.byteswap()
        self.hubble = self.hubble.byteswap()
      
      f.close()
      
    else: # HDF5 snapshot
    
      import h5py
      
      self.format = 3
      
      f = h5py.File(filename,'r')
      
      self.npart = f['/Header'].attrs['NumPart_ThisFile']
      self.massarr = f['/Header'].attrs['MassTable']
      self.time = f['/Header'].attrs['Time']
      self.redshift = f['/Header'].attrs['Redshift']
      self.sfr = f['/Header'].attrs['Flag_Sfr']
      self.feedback = f['/Header'].attrs['Flag_Feedback']
      self.nall = f['/Header'].attrs['NumPart_Total']
      self.cooling = f['/Header'].attrs['Flag_Cooling']
      self.filenum = f['/Header'].attrs['NumFilesPerSnapshot']
      self.boxsize = f['/Header'].attrs['BoxSize']
      self.omega_m = f['/Header'].attrs['Omega0']
      self.omega_l = f['/Header'].attrs['OmegaLambda']
      self.hubble = f['/Header'].attrs['HubbleParam']
      self.swap = 0
      
      f.close()
 
# ----- find offset and size of data block ----- 

def find_block(filename, format, swap, block, block_num, only_list_blocks=False):
  if (not os.path.exists(filename)):
      print( "find_block: file not found ->", filename)
      sys.exit()
            
  f = open(filename,'rb')
  f.seek(0, os.SEEK_END)
  filesize = f.tell()
  f.seek(0, os.SEEK_SET)
  
  found = False
  curblock_num = 1
  while ((not found) and (f.tell()<filesize)):
    if format==2:
      f.seek(4, os.SEEK_CUR)
      curblock = f.read(4)
      if (block == curblock):
        found = True
      f.seek(8, os.SEEK_CUR)  
    else:
      if curblock_num==block_num:
        found = True
        
    curblocksize = (np.fromfile(f,dtype=np.int32,count=1))[0]
    if swap:
      curblocksize = curblocksize.byteswap()
    
    # - print( some debug info about found data blocks -
    #if format==2:
    #  print( curblock, curblock_num, curblocksize
    #else:
    #  print( curblock_num, curblocksize
    
    if only_list_blocks:
      print( curblock_num,curblock,f.tell(),curblocksize)
      found = False
    
    if found:
      blocksize = curblocksize
      offset = f.tell()
    else:
      f.seek(curblocksize, os.SEEK_CUR)
      blocksize_check = (np.fromfile(f,dtype=np.int32,count=1))[0]
      if swap: blocksize_check = blocksize_check.byteswap()
      if (curblocksize != blocksize_check):
        print( "something wrong")
        sys.exit()
      curblock_num += 1
      
  f.close()
      
  if ((not found) and (not only_list_blocks)):
    print( "Error: block not found")
    sys.exit()
    
  if (not only_list_blocks):
    return offset,blocksize
 
# ----- read data block -----
 
def read_block(filename, block, parttype=-1, physical_velocities=True, arepo=0, no_masses=False, verbose=False, cosmic_ray_species=0):
  if (verbose):
      print( "reading block", block)
  
  blockadd=0
  blocksub=0
  
  if arepo==0:
    if (verbose):   
      print( "Gadget format")
    blockadd=0
  if arepo==1:
    if (verbose):   
      print( "Arepo format")
    blockadd=1  
  if arepo==2:
    if (verbose):
      print( "Arepo extended format")
    blockadd=4  
  if no_masses==True:
    if (verbose):   
      print( "No mass block present") 
    blocksub=1
         
  if parttype not in [-1,0,1,2,3,4,5]:
    print( "wrong parttype given")
    sys.exit()
  
  if os.path.exists(filename):
    curfilename = filename
  elif os.path.exists(filename+".0"):
    curfilename = filename+".0"
  elif os.path.exists(filename+".hdf5"):
    curfilename = filename+".hdf5"
  elif os.path.exists(filename+".0.hdf5"):
    curfilename = filename+".0.hdf5"
  else:
    print( "read_block: file not found ->", filename)
    print( "and ->", filename+".0")
    print( "and ->", filename+".hdf5")
    print( "and ->", filename+".0.hdf5")
    sys.exit()
  
  head = snapshot_header(curfilename, cosmic_ray_species=cosmic_ray_species)
  format = head.format
  swap = head.swap
  npart = head.npart
  massarr = head.massarr
  nall = head.nall
  filenum = head.filenum
  redshift = head.redshift
  time = head.time
  del head
  
  if format==3:
    import h5py
  
  # - description of data blocks -
  # add or change blocks as needed for your Gadget version
  data_for_type = np.zeros(6,bool) # should be set to "True" below for the species for which data is stored in the data block
  dt = np.float64 # data type of the data in the block

  hdf5name="none"
  if block=="POS ":
    data_for_type[:] = True
    dt = np.dtype((np.float64,3))
    block_num = 2
    hdf5name = "Coordinates"
  elif block=="VEL ":
    data_for_type[:] = True
    dt = np.dtype((np.float64,3))
    block_num = 3
    hdf5name = "Velocities"
  elif block=="ID  ":
    data_for_type[:] = True
    dt = np.uint32
    block_num = 4
    hdf5name = "ParticleIDs"
  elif block=="MASS":
    data_for_type[np.where(massarr==0)] = True
    block_num = 5
    hdf5name = "Masses"
    if parttype>=0 and massarr[parttype]>0:   
      if (verbose): 
          print( "filling masses according to massarr")
      return np.ones(nall[parttype],dtype=dt)*massarr[parttype]
    elif massarr[np.where(nall>0)[0]].min()>0: # no massblock in snapshot
      data = np.ones(nall.sum(),dtype=dt)
      for i in np.arange(6):
        if i==0:
          startind = 0
        else:
          startind = (nall.cumsum())[i-1] 
        data[startind:(nall.cumsum())[i]] = massarr[i]
      return data
  elif block=="U   ":
    data_for_type[0] = True
    block_num = 6-blocksub
    hdf5name = "InternalEnergy"
  elif block=="RHO ":
    data_for_type[0] = True
    block_num = 7-blocksub
    hdf5name = "Density"
  elif block=="VOL ":
    data_for_type[0] = True
    block_num = 8-blocksub
    hdf5name = "Volume"
  elif block=="CMCE":
    data_for_type[0] = True
    dt = np.dtype((np.float32,3))
    block_num = 9-blocksub 
  elif block=="AREA":
    data_for_type[0] = True
    block_num = 10-blocksub
  elif block=="NFAC":
    data_for_type[0] = True
    dt = np.dtype(np.int32) 
    block_num = 11-blocksub
  elif block=="NE  ":
    data_for_type[0] = True
    block_num = 8+blockadd-blocksub
    hdf5name = "ElectronAbundance"
  elif block=="NH  ":
    data_for_type[0] = True
    block_num = 9+blockadd-blocksub
    hdf5name = "NeutralHydrogenAbundance"
  elif block=="HSML":
    data_for_type[0] = True
    block_num = 10+blockadd-blocksub
    hdf5name = "SmoothingLength"
  elif block=="SFR ":
    data_for_type[0] = True
    block_num = 11+blockadd-blocksub
    hdf5name = "StarFormationRate"
  elif block=="AGE ":
    data_for_type[4] = True
    block_num = 12+blockadd-blocksub
    hdf5name = "GFM_StellarFormationTime"
  elif block=="Z   ":
    data_for_type[0] = True
    data_for_type[4] = True
    block_num = 13+blockadd-blocksub
    hdf5name = "Metallicity"
  elif block=="BHMA":
    data_for_type[5] = True
    block_num = 14+blockadd-blocksub
  elif block=="BHMD":
    data_for_type[5] = True
    block_num = 15+blockadd-blocksub
  elif block=="COOR":
    data_for_type[0] = True
    block_num = -1
  elif block=="DETI":
    data_for_type[0] = True
    block_num = -1
  elif block=="POT ":
    data_for_type[:] = True
    block_num = -1
  elif block=="ACCE":
    data_for_type[:] = True
    dt = np.dtype((np.float64,3))
    block_num = -1
    hdf5name = "Acceleration"
  elif block=="MGPH":
    data_for_type[:] = True
    block_num = -1
    hdf5name = "ModifiedGravityPhi"
  elif block=="MGGP":
    data_for_type[:] = True
    dt = np.dtype((np.float64,3))
    block_num = -1
    hdf5name = "ModifiedGravityGradPhi"
  elif block=="MGAC":
    data_for_type[:] = True
    dt = np.dtype((np.float64,3))
    block_num = -1
    hdf5name = "ModifiedGravityAcceleration"
  else:
    print( "Sorry! Block type", block, "not known!")
    sys.exit()
  # - end of block description -

  if (block_num < 0 and format==1):
    print( "Sorry! Block number of", block, "not known! Unable to read this block from format 1 file!")
    sys.exit() 
  
  if (format==3 and hdf5name=="none"):
    print( "HDF5 name of block", block, "not known! Unable to read this block from format HDF5 snapshot file!")
    sys.exit()
  
  actual_data_for_type = np.copy(data_for_type)  
  if parttype >= 0:
    actual_data_for_type[:] = False
    actual_data_for_type[parttype] = True
    if data_for_type[parttype]==False:
      print( "Error: no data for specified particle type", parttype, "in the block", block) 
      sys.exit()
  elif block=="MASS":
    actual_data_for_type[:] = True  
    
  allpartnum = np.int64(0)
  species_offset = np.zeros(6,np.int64)
  for j in range(6):
    species_offset[j] = allpartnum
    if actual_data_for_type[j]:
      allpartnum += nall[j]
    
  for i in range(filenum): # main loop over files
    if filenum>1:
      curfilename = filename+"."+str(i)
      if format==3:
        curfilename += ".hdf5" 
      
    if i>0:
      head = snapshot_header(curfilename)
      npart = head.npart  
      del head
      
    curpartnum = np.int32(0)
    cur_species_offset = np.zeros(6,np.int64)
    for j in range(6):
      cur_species_offset[j] = curpartnum
      if data_for_type[j]:
        curpartnum += npart[j]
    
    if parttype>=0:
      actual_curpartnum = npart[parttype]      
      add_offset = cur_species_offset[parttype] 
    else:
      actual_curpartnum = curpartnum
      add_offset = np.int32(0)
      
    if format in [1,2]: # ----- read from format 1 or 2 snapshot (none HDF5)-----
      
      offset,blocksize = find_block(curfilename,format,swap,block,block_num)
    
      if i==0: # fix data type for ID if long IDs are used or if double precision is used
        if block=="ID  ":
          if blocksize == np.dtype(dt).itemsize*curpartnum * 2:
            dt = np.uint64       
        elif dt == np.float32:
          if blocksize == np.dtype(dt).itemsize*curpartnum * 2:
            dt = np.float64
        elif dt == (np.float32,3):
          if blocksize == np.dtype(dt).itemsize*curpartnum * 2:
            dt = (np.float64,3)
          
      if np.dtype(dt).itemsize*curpartnum != blocksize:
        print( "something wrong with blocksize! expected =",np.dtype(dt).itemsize*curpartnum,"actual =",blocksize)
        sys.exit()
      
      f = open(curfilename,'rb')
      f.seek(offset + add_offset*np.dtype(dt).itemsize, os.SEEK_CUR)  
      curdat = np.fromfile(f,dtype=dt,count=actual_curpartnum) # read data
      f.close()  
      if swap:
        curdat.byteswap(True)  
        
      if i==0:
        data = np.empty(allpartnum,dt)
      
      for j in range(6):
        if actual_data_for_type[j]:
          if block=="MASS" and massarr[j]>0: # add mass block for particles for which the mass is specified in the snapshot header
            data[species_offset[j]:species_offset[j]+npart[j]] = massarr[j]
          else:
            if parttype>=0:
              data[species_offset[j]:species_offset[j]+npart[j]] = curdat
            else:
              data[species_offset[j]:species_offset[j]+npart[j]] = curdat[cur_species_offset[j]:cur_species_offset[j]+npart[j]]
          species_offset[j] += npart[j]

      del curdat
      
    elif format==3: # ----- read from HDF5 snapshot -----
      
      f = h5py.File(curfilename,'r')
      
      if i==0:
        data = np.empty(allpartnum,dt)
      
      for j in range(6):
        if actual_data_for_type[j]:
          if block=="MASS" and massarr[j]>0: # add mass block for particles for which the mass is specified in the snapshot header
            data[species_offset[j]:species_offset[j]+npart[j]] = massarr[j]
          elif (npart[j]>0 and j!=3):  # Partype 3 = Tracer -> is not physical
            curdat = f["/PartType"+str(j)+"/"+hdf5name].value
            assert data.dtype==curdat.dtype
            data[species_offset[j]:species_offset[j]+npart[j]] = curdat
          species_offset[j] += npart[j]

      f.close()

    else:
      print( "something wrong with format")
      sys.exit()

  if physical_velocities and block=="VEL " and redshift!=0:
    data *= math.sqrt(time)

  return data
  
# ----- list all data blocks in a format 2 snapshot file -----

def list_format2_blocks(filename):
  if (not os.path.exists(filename)):
      print( "list_format2_blocks: file not found ->", filename)
      sys.exit()
  
  head = snapshot_header(filename)
  format = head.format
  swap = head.swap
  del head
  
  if (format != 2):
    print( "not a format 2 snapshot file")
    sys.exit()
            
  print( "#   BLOCK   OFFSET   SIZE")
  print( "-------------------------")
  
  find_block(filename, format, swap, "XXXX", 0, only_list_blocks=True)
  
  print( "-------------------------")
