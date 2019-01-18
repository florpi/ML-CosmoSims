
import numpy as np
import h5py
from mpi4py import MPI
import glob, os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numba
from numba import jit
import time
import create_files as cf


#******************** SIMULATION INFO ********************# 

simulation_path = '/cosma5/data/dp004/dc-cues1/TNG300-1/'
particles_path = simulation_path + 'snapdir_099/'
group_path = simulation_path + 'groups_099/'
save_path = '/cosma6/data/dp004/dc-cues1/environment/TNG300-1/'
data_path = particles_path

# If it exists, read file that contains bounding boxes for every particle file
if not os.path.isfile(data_path + 'filelimits.txt'):
	cf.file_limits(particles_path, data_path)	
xmin,xmax,ymin,ymax,zmin,zmax = np.loadtxt(data_path + 'filelimits.txt', unpack=True)

# Read halo files:
if not os.path.isfile(data_path + 'group_info.npy'):
	cf.halo_info(group_path, data_path)	

halos = np.load(data_path + 'group_info.npy')
halopos = halos[:,:3]
halor200c = halos[:,-1]

print('All needed info saved')

#******************** FUNCTIONS TO DEFINE FILE INTERSECTIONS ********************# 

def sphere(center, radius):
	'''
	Defines sphere
		Args:
			center
			radius
		Returns:
			x, y, z, coordinates of the sphere
	'''

	u = np.linspace(0, 2*np.pi, 100)
	v = np.linspace(0, np.pi, 100)
	x = center[0] + radius * np.outer(np.cos(u),np.sin(v))
	y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
	z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
	return x,y,z



def doesCubeIntersectSphere(xmin,xmax,ymin,ymax,zmin,zmax,center_sphere,radius):
	'''
	Checks if a given sphere intersects a cube.
		Args:
			xmin, xmax, ymin, ymax, zmin, zmax, coordinates definining the box
			center_sphere, center of the sphere
			radius, radius of the sphere
		Returns:
			Bool, true if they intersect false if not
	'''

	dist_squared = radius**2
	if(center_sphere[0] < xmin):
		dist_squared -= (center_sphere[0] - xmin)**2
	elif(center_sphere[0] > xmax):
		dist_squared -= (center_sphere[0] - xmax)**2
	if(center_sphere[1] < ymin):
		dist_squared -= (center_sphere[1] - ymin)**2
	elif(center_sphere[1] > ymax):
		dist_squared -= (center_sphere[1] - ymax)**2
	if(center_sphere[2] < zmin):
		dist_squared -= (center_sphere[2] - zmin)**2
	elif(center_sphere[2] > zmax):
		dist_squared -= (center_sphere[2] - zmax)**2
			
	return dist_squared > 0



particles_files = glob.glob(particles_path + "*.hdf5")
numbers = [int(element.split('/')[-1].split('.')[1])            for element in particles_files]
numbers, particles_files = (list(t) for t in zip(*sorted(zip(numbers, 
                                         particles_files))))


@jit(nopython=True)
def compute_nparticles(v1,v2,radius,bins):
	'''
	Computes the number of particles at distances given by bins from the halo position
		Args:
			v1, array of particle's positions
			v2, array of halo positions
			radius, r200c of the halo 
			bins, array of bins with limits where to compute the number of particles
		Returns:
			nparticles, array containing number of particles within bins
			r_centers, centers of the bins
	'''
	distances = np.sqrt(np.sum((v1-v2)**2,axis=-1))/radius
	nparticles = len(np.where(distances < (50))[0])
	counts, bins = np.histogram(distances, bins)
	r_centers = 0.5 * (bins[1:] + bins[:-1])
	#volume = 4./3. * np.pi * (bins[1:]**3 - bins[:-1]**3)
	#density = counts / volume # number density
	return counts, r_centers


dm = 'PartType1'

# bin particles 
innerbins = np.logspace(-2.5, 0.,32)
outerbins = np.array([2, 20 , 50])
bins = np.concatenate((innerbins,outerbins))
Nhalos = halor200c.shape[0] 
Nparts = np.zeros((Nhalos, len(bins)-1)) # Nhalos x Nbins
r_centers = 0.5 * (bins[1:] + bins[:-1])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print('Computing density using %d processors'%size)



numDataPerRank = int(len(particles_files)/size)
data = None
if rank == 0:
	data = np.arange(len(particles_files), dtype='i')

recvbuf = np.empty(numDataPerRank, dtype='i')

    
comm.Scatter(data, recvbuf, root=0)

recvbuf = np.asarray(recvbuf).astype(int)
particles_files = np.asarray(particles_files)

print('Rank :',rank, ' , files : ', particles_files[recvbuf])

'''
if type(particles_files[recvbuf]) is not list:
	particles_loop = [particles_files[recvbuf]]
else:
	particles_loop = particles_files[recvbuf]
'''
particles_loop = particles_files[recvbuf]

for (i,file) in enumerate(particles_loop):
	with h5py.File(file, 'r') as hf:
		# Particle coordinates in that file
		coordinates = hf[dm + '/Coordinates'][:]
		for j in range(Nhalos):

			intersection = doesCubeIntersectSphere(xmin[recvbuf[i]],xmax[recvbuf[i]],
					ymin[recvbuf[i]],ymax[recvbuf[i]],zmin[recvbuf[i]],zmax[recvbuf[i]],
					halopos[j,:],50*halor200c[j])
			if(intersection):

				nparticles, r_center = compute_nparticles(coordinates,halopos[j,:],halor200c[j],bins)  
				Nparts[j,:] = nparticles 
		np.save(save_path + 'nparticles_snapfile%d_processor%i'%(i, comm.rank), Nparts)

