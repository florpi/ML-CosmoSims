# Create dataset
#
from __future__ import division
import os, sys
import numpy as np
import pandas as pd
import read_hdf5
import h5py
from itertools import product

args = {}
args["simdir"]   = sys.argv[1]
args["outdir"]   = sys.argv[2]
args["nsnap"]    = sys.argv[3]
args["nvoxel"]   = sys.argv[4]

s = read_hdf5.snapshot(args["nsnap"], args["simdir"]) 
s.read(["Coordinates"], parttype=[1])
dm_pos = (s['Coordinates']['dm']).astype('float64')
print(np.max(dm_pos[:,0]))

dmdf=pd.DataFrame({
    'x_vox' : (np.floor((dm_pos[:, 0]/62000/args["nvoxel"]))).astype(int),
    'y_vox' : (np.floor((dm_pos[:, 1]/62000/args["nvoxel"]))).astype(int),
    'z_vox' : (np.floor((dm_pos[:, 2]/62000/args["nvoxel"]))).astype(int),
})

mergetable['c']=1                                                            
def dataframe_to_array(df, out_shp):                                         
    ids = np.ravel_multi_index(df[['x_vox','y_voxb','z_vox']].values.T, out_shp)    
    val = df['count'].values                                                 
    return np.bincount(ids, val, minlength=np.prod(out_shp)).reshape(out_shp)

counts=mergetable.groupby(['x_vox','y_vox','z_vox'])['c'].count().reset_index(name="count")
arr=dataframe_to_array(counts, (1024,1024,1024))                             
old_arr=np.load('/scratch/xz2139/Dark_zeros.npy')                            
np.save('/scratch/xz2139/Dark_zeros.npy',old_arr+arr)                        
print(name) 

