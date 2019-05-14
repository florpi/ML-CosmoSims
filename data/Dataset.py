from torch.utils import data
import numpy as np


class Dataset(data.Dataset):
    """
    Class to control dataset input into ML-algorithm.
    """

    def __init__(
        self,
        catalogue_ids,
        features_file,
        targets_file,
        cat="count",
        vel=False,
        aug=False,
        reg=False,
        normalize=False,
    ):
        """
        Initialization

        Parameters
        ----------
        data_path : str
            The path to the file in which data-set is found
        label_path : str
            The path to the file in which label-set is found
        cat : str
            Define which target data to use. Can be either 'mass' or 'count'
        vel : boolean
            Switch to use object velocity or not.
        aug : boolean
            Switch for data augmentation or not.
        reg : boolean
            ???
        """
        self.ids = catalogue_ids
        self.features_file = features_file
        self.targets_file = targets_file
        self.target = cat
        self.aug = aug
        self.reg = reg
        self.vel = vel
        self.normalize = normalize

    def __getitem__(
            self,
            index,
    ):
        """
        Parameters
        ----------
        index : list
            Coordinates of the voxels in the data-set
        **params : dict
            batch-size, shuffle, num_workers
        
        Returns
        -------
        feature_boxes : np.ndarray
            Dark-matter input
        target_boxes : np.ndarray
            Target values e.g. single stars or galaxies
        """
        # Load dark-matter catalogue
        feature_boxes = np.load(self.features_file)  #3D
        feature_boxes = [feature_boxes[xx] for xx in self.ids]
        feature_boxes = np.expand_dims(feature_boxes, axis=0)  #4D

        if self.normalize is True:
            # Normalize dark-matter data w.r.t. maximum
            feature_boxes = feature_boxes / np.max(feature_boxes)

        if self.target == "count":
            # Use nr. of stars as target
            target_boxes = np.load(self.targets_file)
            target_boxes = [target_boxes[xx] for xx in self.ids]
            if not self.reg:
                # Convert python function to vector function
                convert = np.vectorize(self.convert_class)
                target_boxes = convert(target_boxes)

        elif self.target == "mass":
            # use mass as target
            target_boxes = np.load(self.targets_file)

        if self.aug is True:
            # Dataset augmentation
            dim_to_flip = tuple(
                np.arange(3)[np.random.choice(a=[False, True], 
                size=3)]
            )
            if len(dim_to_flip) > 0:
                dmboxes = np.flip(feature_boxes, dim_to_flip)
                target_boxes = np.flip(target_boxes, dim_to_flip)

        #if self.vel is True:
        #    # Add velocity info. to dm-dataset
        #    veloset = np.load()
        #    dmboxes = np.concatenate((dmboxes, veloset), axis=0)

        return feature_boxes, target_boxes

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.ids)

    def convert_class(self, num):
        if num==0:
            return 0
        elif num>0:
            return 1
        else:
            print('dark matter mass smaller than 0')
