import h5py
import numpy as np

filename = 'test.hdf5'

f = h5py.File(filename, 'r')


print(f['0'].shape)
print(f['2'][:])
