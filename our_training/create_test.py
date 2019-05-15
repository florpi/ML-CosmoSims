import h5py
import numpy as np

n_samples = 4
woxel_ids = np.arange(0,n_samples)

n_particles = np.random.random(size = (20, 32, 32,32))

f = h5py.File("test.hdf5", "w")

for vi in woxel_ids:
	f.create_dataset('id-' + str(vi+1), data = n_particles[vi])


