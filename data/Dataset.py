from torch.utils import data
import numpy as np


class Dataset(data.Dataset):
    """
    Class to control dataset input into neural network.
    """

    def __init__(
        self,
        data_path,
        data_partition,
        cosmo_type,
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
            The path to the directory in which dataset is found
        data_partition : str
            Indicates whether to use training, testing, or validating
        cosmo_type : str
            Indicates whether to use dark-matter-only or full-physics simulations
        cat : str
            Define which target data to use. Can be either 'mass' or 'count'
        vel : boolean
            Switch to use object velocity or not.
        aug : boolean
            Switch for data augmentation or not.
        reg : boolean
            ???
        """
        self.data_path = data_path
        self.data_partition = data_partition
        self.cosmo_type = cosmo_type
        self.target = cat
        self.aug = aug
        self.reg = reg
        self.vel = vel
        self.normalize = normalize

    def __getitem__(self, index):
        """
        Returns
        -------
        dmset : np.ndarray
            Dark-matter input
        labels : np.ndarray
            Target values e.g. single stars or galaxies
        """
        # Load dark-matter catalogue
        dmset = np.load(
            self.datapath
            + "/"
            + data_partition
            + "/"
            + self.cosmo_type
            + "/"
            + self.cosmo_type
            + "_dm.npy"
        )
        dmset = np.expand_dims(dmset, axis=0)

        if self.normalize is True:
            # Normalize dark-matter data w.r.t. maximum
            dmset = dmset / np.max(dmset)

        if self.cat == "count":
            # Use nr. of stars as target
            labels = np.load(
                self.datapath
                + "/"
                + data_partition
                + "/"
                + self.cosmo_type
                + "/"
                + self.cosmo_type
                + "_st.npy"
            )
            if not self.reg:
                # Convert python function to vector function
                convert = np.vectorize(self.convert_class)
                lables = convert(labels)
        elif self.cat == "mass":
            # use mass as target
            labels = np.load(
                self.datapath
                + "/"
                + data_partition
                + "/"
                + self.cosmo_type
                + "/"
                + self.cosmo_type
                + "_st.npy"
            )

        if self.aug is True:
            # Dataset augmentation
            dim_to_flip = tuple(np.arange(3)[np.random.choice(a=[False, True], size=3)])
            if len(dim_to_flip) > 0:
                dmsets = np.flip(dmsets, dim_to_flip)
                labels = np.flip(labels, dim_to_flip)

        if self.vel is True:
            # Add velocity info. to dm-dataset
            veloset = np.load(
                self.datapath
                + "/"
                + data_partition
                + "/"
                + self.cosmo_type
                + "/"
                + self.cosmo_type
                + "_dm.npy"
            )
            dmset = np.concatenate((dmset, veloset), axis=0)
        return dmset, labels

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.dataset)
