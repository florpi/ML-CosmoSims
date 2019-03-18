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
s.read(["Coordinates"], parttype=[0])
print(type(sdata['Coordinates']['dm']))


dm_pos = (sdata['Coordinates']['dm']*scale).astype('float64')

stable=pd.DataFrame(dm_pos)
postable.columns=(['x','y','z'])
postable['x_b']=np.floor(postable['x']/(75000/1024)) 
postable['y_b']=np.floor(postable['y']/(75000/1024))
postable['z_b']=np.floor(postable['z']/(75000/1024))
postable['x_b'] = postable['x_b'].astype(int)
postable['y_b'] = postable['y_b'].astype(int)
postable['z_b'] = postable['z_b'].astype(int)

mergetable['c']=1                                                            
def dataframe_to_array(df, out_shp):                                         
    ids = np.ravel_multi_index(df[['x_b','y_b','z_b']].values.T, out_shp)    
    val = df['count'].values                                                 
    return np.bincount(ids, val, minlength=np.prod(out_shp)).reshape(out_shp)

counts=mergetable.groupby(['x_b','y_b','z_b'])['c'].count().reset_index(name="count")
arr=dataframe_to_array(counts, (1024,1024,1024))                             
old_arr=np.load('/scratch/xz2139/Dark_zeros.npy')                            
np.save('/scratch/xz2139/Dark_zeros.npy',old_arr+arr)                        
print(name) 

x_out = Pa['Pos'][binds, 0]
y_out = Pa['Pos'][binds, 1]
z_out = Pa['Pos'][binds, 2]
