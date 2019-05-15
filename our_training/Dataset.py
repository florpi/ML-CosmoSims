from torch.utils import data
import numpy as np
import h5py

#based on https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

class Dataset(data.Dataset):

	def __init__(self, list_IDs, labels):

		self.labels = labels 
		self.list_IDs = list_IDs

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)


	def __getitem__(self, index):
		'Generates one sample of data'
		# Select sample
		ID = self.list_IDs[index]

		# Load data and get label
		#X = torch.load('data/' + ID + '.pt')
		X = h5py.File(data_filename)[str(ID)][:]
		y = self.labels[ID]

		return X, y


