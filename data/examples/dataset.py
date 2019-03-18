import numpy as np

class Dataset(data.Dataset):
    def __init__(self, lists):
        'Initialization'
        self.IDs = lists

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.IDs[index]
        d_box=np.load('/scratch/xz2139/cosmo_dark/arrays/'+str(ID[0])+'_'+str(ID[1])+'_'+str(ID[2])+'.npy')
        f_box=np.load('/scratch/xz2139/cosmo_full/arrays/'+str(ID[0])+'_'+str(ID[1])+'_'+str(ID[2])+'.npy')
        return d_box,f_box