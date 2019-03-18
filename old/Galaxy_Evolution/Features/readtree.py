#!/usr/bin/env python3
import logging
import os
import h5py
import numpy as np
import pandas as pd
from collections import Counter

logging.basicConfig(level=logging.DEBUG)


def read_tree_keys(fname):
    df = h5py.File(fname, 'r')
    print('The following merger-tree keys are available:')
    for key in list(df.keys()):
        print(':%s has %d sub-keys' % (key, len(list(df[key].keys()))))
        if len(list(df[key].keys())) > 0:
            for subkey in df[key].keys():
                print('\t ::%s' % subkey)
    print('\n')


class DHaloReader():
    """DHalo Reader class.
    """
    def __init__(self, fname, simtype):
        if simtype == 'original':
            self.filename = fname
            self.columns = [
                "nodeIndex",
                "snapshotNumber",
                "hostIndex",
                "descendantHost",
                "descendantIndex",
                "isMainProgenitor",
                "mainProgenitorIndex",
                "isInterpolated"
            ]
            self.data = self.read(simtype)
        elif simtype == 'EAGLE':
            self.filename = fname
            self.columns = [
                "DhaloIndex",
                "GroupIndex",
                "SnapNum",
                "DescendantID",
                "HaloID",
                "LastProgID",
                "TopLeafID"
            ]
            self.data = self.read(simtype)

    def read(self, simtype):
        """Reads DHalo data into memory

        Output data format:

        ===========  ==================
         Column ID    Column Name
        ===========  ==================
                 0    nodeIndex
                 1    hostIndex
                 2    descendantHost
                 3    descendantIndex
                 4    isMainProgenitor
                 5    mainProgenitorIndex
                 6    isInterpolated
        ===========  ==================

        nodeIndex:
            index of each halo or subhalo, unique across the entire catalogue
        descendantIndex:
            index of a descendanta halo (if multiple haloes have the same
            descendatant index, they all are the progenitors)
        snapshotNumber:
            snapshot at which halo was identified
        particleNumber:
            number of particles in a halo; might differ from masses identified
            by other methods
        hostIndex:
            index of a host halo; for subhaloes, this points to a parent halo;
            for main haloes, this points to themselves
        descendantHost:
            index of a host halo of descendant of a halo (or subhalo); this
            field eliminates "multiple descendance" problem, always creating a
            merger history which works for main progenitors only
        isMainProgenitor:
            1 if it is
        """

        if simtype == 'original':
            if self.filename.endswith(".pkl"):
                logging.debug("Loading pickle file %s", self.filename)
                data = pd.read_pickle(self.filename)

            elif self.filename.endswith(".hdf5"):
                logging.debug("Loading HDF5 file %s", self.filename)
                with h5py.File(self.filename, "r") as data_file:
                    #print('treeIndex', data_file["treeIndex"].keys())
                    #print('haloTrees', data_file["haloTrees"].keys())
                    
                    # Find dimensionality of keys
                    columns_1dim = [] 
                    columns_2dim = [] 
                    for column in self.columns:
                        if len(data_file["/haloTrees/%s" % column].shape) == 1:
                            columns_1dim.append(column)
                        else:
                            columns_2dim.append(column)
                    
                    # 1D keys
                    data = pd.DataFrame(
                        {
                            column: data_file["/haloTrees/%s" % column].value
                            for column in columns_1dim
                        },
                        columns=columns_1dim
                    ).set_index("nodeIndex")
                    del columns_1dim

                    # 2D keys
                    for column in columns_2dim:
                        if column == 'position':
                            pos = data_file["/haloTrees/%s" % column].value
                            data['X'] = pd.Series(pos[:, 0], index=data.index)
                            data['Y'] = pd.Series(pos[:, 1], index=data.index)
                            data['Z'] = pd.Series(pos[:, 2], index=data.index)
                    del columns_2dim

                    data.rename(index=str,
                                columns={"snapshotNumber": "snapnum"})
                    ## eliminate fake elements with isIntegrated=1
                    #data = data[data.isInterpolated != 1]

            else:
                raise TypeError("Unknown filetype %s" % self.filename)
        if simtype == 'EAGLE':
            if self.filename.endswith(".pkl"):
                logging.debug("Loading pickle file %s", self.filename)
                data = pd.read_pickle(self.filename)

            elif self.filename.endswith(".hdf5"):
                logging.debug("Loading HDF5 file %s", self.filename)
                data_file = h5py.File(self.filename, 'r')
                column_mt = []
                column_sh = []
                for column in self.columns:
                    if column in data_file['MergerTree']:
                        column_mt.append(column)
                    else:
                        column_sh.append(column)

                data = pd.DataFrame(
                    {
                        column: data_file["/MergerTree/%s" % column].value
                        for column in column_mt
                    },
                    columns=column_mt
                ).set_index("HaloID")
                #.set_index(data_file["/Merger/HaloID"].value)

                for column in column_sh:
                    data[column] = pd.Series(data_file["/Subhalo/%s" % column].value,
                                             index=data.index)
                data = data.rename(index=str,
                                   columns={"SnapNum": "snapnum", #"HaloID": "nodeIndex",
                                            "DescendantID" : "descendantIndex"})
            else:
                raise TypeError("Unknown filetype %s" % self.filename)

        return data


    def get_halo(self, index):
        """Returns halo (row of data) given a ``nodeIndex``

        :param int index: ``nodeIndex`` queried
        :return numpy.ndarray: row of argument ``d`` of the given ``nodeIndex``
        """
        try:
            halo = self.data.loc[index]
        except KeyError:
            raise IndexError(
                "Halo id %d not found in %s" % (index, self.filename)
            )
        return halo


    def halo_progenitor_ids(self, index):
        """Finds indices of all progenitors of a halo, recursively.

        The following search is employed:

        - find all haloes of which 'h' is a **host of a descendant**
        - find hosts of **these haloes**
        - keep unique ones
        """
        _progenitors = []

        def rec(i):
            _progenitor_ids = self.data[self.data["descendantHost"] == i][
                "hostIndex"
            ].unique()

            logging.debug("Progenitors recursion: %d > %d (%d progenitors)",
                          index,
                          i,
                          len(_progenitor_ids))
            
            if len(_progenitor_ids) == 0:
                return
           
            for _progenitor_id in _progenitor_ids:
                # if _progenitor_id not in _progenitors:
                # TODO: this only eliminates fly-byes
                _progenitors.append(_progenitor_id)
                rec(_progenitor_id)

        rec(index)

        logging.info(
            "%d progenitors found for halo %d", len(_progenitors), index
        )
        return _progenitors


    def halo_host(self, index):
        """Finds host of halo.

        Recursively continues until hits the main halo, in case of multiply embedded
        subhaloes.
        """
        halo = self.get_halo(index)
        return (
            halo
            if halo.name == halo["hostIndex"]
            else self.halo_host(self.get_halo(halo["hostIndex"]).name)
        )


    def halo_mass(self, index):
        """Finds mass of central halo and all subhaloes.
        """
        return self.data[self.data["hostIndex"] == index][
            "particleNumber"
        ].sum()


    def collapsed_mass_history(self, index, nfw_f):
        """Calculates mass assembly history for a given halo.

        Tree-based approach has been abandoned for performace reasons.
        
        Input:
            int index: nodeIndex
            float nfw_f: NFW :math:`f` parameter
        Output:
            numpy.ndarray: CMH with rows formatted like ``[nodeIndex,
                           snapshotNumber, sum(particleNumber)]``
        """

        logging.debug("Looking for halo %d", index)
        halo = self.get_halo(index)
        if halo["hostIndex"] != halo.name:
            raise ValueError("Not a host halo!")
        m_0 = self.halo_mass(index)

        progenitors = pd.concat(
            [
                self.data.loc[index],
                self.data.loc[self.halo_progenitor_ids(index)],
            ]
        )
        logging.debug(
            "Built progenitor sub-table for halo %d of mass %d with %d members",
            index,
            m_0,
            progenitors.size,
        )

        progenitors = progenitors[progenitors["particleNumber"] > nfw_f * m_0]
        cmh = progenitors.groupby("snapshotNumber", as_index=False)[
            "particleNumber"
        ].sum()
        cmh["nodeIndex"] = index
        logging.info(
            "Aggregated masses of %d valid progenitors of halo %d",
            progenitors.size,
            index,
        )

        return cmh

    
    def tot_num_of_progenitors_at_z(self, SH, mtree, z1, z2):
        """
        Find the all top-nodes mtree(z1) (across different snapshots) of merger-trees
        with base-node in SH(z2)
        
        Input:
            SH: re-ordered SubFind halo library with subhalos at chosen redshift
            mtree: DHalo merger-tree library
            z1: starting redshift where top-nodes are searched
            z2: final redshift where merger-tree base exists
        Output:
            nodeID: nodeID's at z2
            progcounts: nr. of progentiros at z1
        """
        
        for ss in range(z1, z2+1):
            print('redshift:', ss)
            # nodes at redshift ss
            ss_indx = np.where(mtree.data.snapshotNumber.values == ss)
            nodeID = mtree.data.index.values[ss_indx]
            nodeID_desc = mtree.data.descendantIndex.values[ss_indx]
            
            # find number of progenitors for nodes at redshift ss
            if ss != z1:
                progcounts = np.zeros(len(nodeID), dtype=int)
                for ii in range(len(nodeID_past_desc)):
                    if nodeID_past_desc[ii] in nodeID:
                        indx = np.where(nodeID == nodeID_past_desc[ii])
                        progcounts[indx] = count[ii]

            nodeID_desc_unique, count = np.unique(nodeID_desc, return_counts=True)
            nodeID_desc_unique=nodeID_desc_unique[1:]; count=count[1:]
            
            # add progenitors of progenitors
            if ss != z1:
                for ii in range(len(nodeID)):
                    if progcounts[ii] > 1:
                        indx = np.where(nodeID_desc_unique == nodeID_desc[ii])
                        count[indx] += progcounts[ii] - 1

            nodeID_past = nodeID
            nodeID_past_desc = nodeID_desc_unique
        return nodeID, progcounts


    def find_progenitors_at_z(self, SH, mtree, z1, z2):
        """
        Number of progenitors at one given snapshot z1

        Input:
            SH: re-ordered SubFind halo library with subhalos at chosen redshift
            mtree: DHalo merger-tree library
            z: final redshift/end of tree
        """
        
        for ss in range(z1, z2):
            # nodes at redshift ss
            ss_indx = np.where(mtree.data.snapshotNumber.values == ss)
            nodeID = mtree.data.index.values[ss_indx]
            nodeID_desc = mtree.data.descendantIndex.values[ss_indx]
            
            # find number of progenitors for nodes at redshift ss
            if ss != z1:
                _progcounts = np.zeros(len(nodeID))
                for ii in range(len(nodeID_past_desc)):
                    if nodeID_past_desc[ii] in nodeID:
                        indx = np.where(nodeID == nodeID_past_desc[ii])
                        _progcounts[indx] = count[ii]

            nodeID_desc_unique, count = np.unique(nodeID_desc, return_counts=True)
            nodeID_desc_unique=nodeID_desc_unique[1:]; count=count[1:]
            
            nodeID_past = nodeID
            nodeID_past_desc = nodeID_desc_unique
            if ss != z1:
                _progcounts_past = _progcounts
        print('_progcounts', _progcounts)
    
    
    def find_progenitors_until_z(self, mtree, nodeID, z1, z2):
        """
        Number of progenitors att all redshift between z1 and z2.

        Input:
            mtree: DHalo merger-tree library
            z1: final redshift/end of tree
            z2: redshift of observed subhalo/-galaxy
        Output:
            nodeID: nodeID's of structure at z2
            progcounts: 2D list of progcounts per snapshot
        """
        snapcount = 0
        print('from %d until %d' % (z2, z1))
        for ss in range(z2, z1, -1):
            if ss == z2:
                df_target = pd.DataFrame({'nodeID':nodeID})
                _indx = np.where(mtree.data.snapshotNumber.values == ss-1)
                nodeID_prog = mtree.data.index.values[_indx]
                nodeID_prog_desc = mtree.data.descendantIndex.values[_indx]
                _indx = np.where((nodeID_prog_desc < 1e15) &
                                 (nodeID_prog_desc > 1e11))
                nodeID_prog = nodeID_prog[_indx]
                nodeID_prog_desc = nodeID_prog_desc[_indx]

                df_prog = pd.DataFrame({'nodeID' : nodeID_prog,
                                        'nodeID_target' : nodeID_prog_desc})

                # Initiliaze Output Array
                progcounts = np.zeros((df_target['nodeID'].size, z2-z1))

                # nodeID_prog_desc_unic is sorted
                nodeID_prog_desc_unic, count = np.unique(nodeID_prog_desc,
                                                         return_counts=True)
                # remove -1's
                nodeID_prog_desc_unic=nodeID_prog_desc_unic[1:]; count=count[1:]

                # Nr. of progenitors for sub-&halos at snapshot z2
                s = pd.Index(df_target['nodeID'].tolist())
                _indx_now = s.get_indexer(list(nodeID_prog_desc_unic))
                now_sort_indx = np.argsort(df_target['nodeID'].values[_indx_now])
                pro_sort_indx = np.argsort(nodeID_prog_desc_unic)
                progcounts[_indx_now[now_sort_indx], snapcount] = count[pro_sort_indx]
                    
            else:
                df_now = df_prog
                _indx = np.where(mtree.data.snapshotNumber.values == ss-1)
                nodeID_prog = mtree.data.index.values[_indx]
                nodeID_prog_desc = mtree.data.descendantIndex.values[_indx]
                #_indx = np.where((nodeID_prog_desc < 1e15) &
                #                 (nodeID_prog_desc > 1e10))
                #nodeID_prog = nodeID_prog[_indx]
                #nodeID_prog_desc = nodeID_prog_desc[_indx]
                df_prog = pd.DataFrame({'nodeID' : nodeID_prog})
         
                progcounts_local = np.zeros(df_now['nodeID'].size)
                nodeID_prog_desc_unic, count = np.unique(nodeID_prog_desc,
                                                         return_counts=True)
                # remove -1's
                nodeID_prog_desc_unic=nodeID_prog_desc_unic[1:]; count=count[1:]
                
                # progenitors for snapshot ss
                s = pd.Index(df_now['nodeID'].tolist())
                _indx_now = s.get_indexer(list(nodeID_prog_desc_unic))
                now_sort_indx = np.argsort(df_now['nodeID'].values[_indx_now])
                pro_sort_indx = np.argsort(nodeID_prog_desc_unic)
                progcounts_local[_indx_now[now_sort_indx]] = count[pro_sort_indx]
                df_now['progcount'] = pd.Series(progcounts_local,
                                                index=df_now.index, dtype=int)

                # Nr. of progenitors for sub-&halos at snapshot z2
                df_inter = df_now.groupby(['nodeID_target'],
                                          as_index=False)['progcount'].sum()
                # only real progeniteurs
                df_inter = df_inter[(df_inter['nodeID_target'] > 1e10) & 
                                    (df_inter['nodeID_target'] < 1e15)]
                df_inter = df_inter.drop_duplicates(subset=['nodeID_target'],
                                                    keep='first')
                
                s = pd.Index(df_target['nodeID'].tolist())
                _indx_now = s.get_indexer(df_inter['nodeID_target'].tolist())
                now_sort_indx = np.argsort(df_target['nodeID'].values[_indx_now])
                pro_sort_indx = np.argsort(df_inter['nodeID_target'].values)
                progcounts[_indx_now[now_sort_indx], snapcount] = df_inter['progcount'].values[pro_sort_indx]

                # sort nodeID_prog to nodeID
                #s = pd.Index(df_now['nodeID'].tolist())
                #_indx_now = s.get_indexer(list(nodeID_prog_desc_unic))
                #df_now['nodeID_target'].values[_indx_now]
                
                obs_ref_local = np.zeros(df_prog['nodeID'].size)
                for ii in range(len(nodeID_prog_desc_unic)):
                    tarID = df_now.loc[
                            df_now['nodeID'] == nodeID_prog_desc_unic[ii],
                            'nodeID_target'].values.astype(int)
                    if tarID:
                        _indx = np.where(
                                nodeID_prog_desc == nodeID_prog_desc_unic[ii])
                        obs_ref_local[_indx] = tarID
                df_prog['nodeID_target'] = pd.Series(obs_ref_local,
                                                     index=df_prog.index)

            snapcount += 1
        del nodeID_prog_desc
        del df_now, df_inter, df_prog
        return np.asarray(df_target['nodeID'].tolist()), progcounts


    def find_progenitors_until_z_EAGLE(self, mtree, nodeID, z1, z2):
        """
        Number of progenitors att all redshift between z1 and z2.

        Input:
            mtree: DHalo merger-tree library
            z1: final redshift/end of tree
            z2: redshift of observed subhalo/-galaxy
        Output:
            nodeID: nodeID's of structure at z2
            progcounts: 2D list of progcounts per snapshot
        """
        snapcount = 0
        print(':Read MergerTree from %d until %d' % (z2, z1))
        for ss in range(z2, z1, -1):
            if ss == z2:
                df_target = pd.DataFrame({'nodeID':nodeID})
                _indx = np.where(mtree.data.snapnum.values == ss-1)
                nodeID_prog = mtree.data.index.values[_indx]
                nodeID_prog_desc = mtree.data.descendantIndex.values[_indx]
                _indx = np.where((nodeID_prog_desc < 1e15) &
                                 (nodeID_prog_desc > 1e11))
                nodeID_prog = nodeID_prog[_indx]
                nodeID_prog_desc = nodeID_prog_desc[_indx]

                df_prog = pd.DataFrame({'nodeID' : nodeID_prog,
                                        'nodeID_target' : nodeID_prog_desc})

                # Initiliaze Output Array
                progcounts = np.zeros((df_target['nodeID'].size, z2-z1))

                # nodeID_prog_desc_unic is sorted
                nodeID_prog_desc_unic, count = np.unique(nodeID_prog_desc,
                                                         return_counts=True)
                # remove -1's
                nodeID_prog_desc_unic=nodeID_prog_desc_unic[1:]; count=count[1:]

                # Nr. of progenitors for sub-&halos at snapshot z2
                s = pd.Index(df_target['nodeID'].tolist())
                _indx_now = s.get_indexer(list(nodeID_prog_desc_unic))
                now_sort_indx = np.argsort(df_target['nodeID'].values[_indx_now])
                pro_sort_indx = np.argsort(nodeID_prog_desc_unic)
                progcounts[_indx_now[now_sort_indx], snapcount] = count[pro_sort_indx]
                    
            else:
                df_now = df_prog
                _indx = np.where(mtree.data.snapnum.values == ss-1)
                nodeID_prog = mtree.data.index.values[_indx]
                nodeID_prog_desc = mtree.data.descendantIndex.values[_indx]
                #_indx = np.where((nodeID_prog_desc < 1e15) &
                #                 (nodeID_prog_desc > 1e10))
                #nodeID_prog = nodeID_prog[_indx]
                #nodeID_prog_desc = nodeID_prog_desc[_indx]
                df_prog = pd.DataFrame({'nodeID' : nodeID_prog})
         
                progcounts_local = np.zeros(df_now['nodeID'].size)
                nodeID_prog_desc_unic, count = np.unique(nodeID_prog_desc,
                                                         return_counts=True)
                # remove -1's
                nodeID_prog_desc_unic=nodeID_prog_desc_unic[1:]; count=count[1:]
                
                # progenitors for snapshot ss
                s = pd.Index(df_now['nodeID'].tolist())
                _indx_now = s.get_indexer(list(nodeID_prog_desc_unic))
                now_sort_indx = np.argsort(df_now['nodeID'].values[_indx_now])
                pro_sort_indx = np.argsort(nodeID_prog_desc_unic)
                progcounts_local[_indx_now[now_sort_indx]] = count[pro_sort_indx]
                df_now['progcount'] = pd.Series(progcounts_local,
                                                index=df_now.index, dtype=int)

                # Nr. of progenitors for sub-&halos at snapshot z2
                df_inter = df_now.groupby(['nodeID_target'],
                                          as_index=False)['progcount'].sum()
                # only real progeniteurs
                df_inter = df_inter[(df_inter['nodeID_target'] > 1e10) & 
                                    (df_inter['nodeID_target'] < 1e15)]
                df_inter = df_inter.drop_duplicates(subset=['nodeID_target'],
                                                    keep='first')
                
                s = pd.Index(df_target['nodeID'].tolist())
                _indx_now = s.get_indexer(df_inter['nodeID_target'].tolist())
                now_sort_indx = np.argsort(df_target['nodeID'].values[_indx_now])
                pro_sort_indx = np.argsort(df_inter['nodeID_target'].values)
                progcounts[_indx_now[now_sort_indx], snapcount] = df_inter['progcount'].values[pro_sort_indx]

                # sort nodeID_prog to nodeID
                #s = pd.Index(df_now['nodeID'].tolist())
                #_indx_now = s.get_indexer(list(nodeID_prog_desc_unic))
                #df_now['nodeID_target'].values[_indx_now]
                
                obs_ref_local = np.zeros(df_prog['nodeID'].size)
                for ii in range(len(nodeID_prog_desc_unic)):
                    tarID = df_now.loc[
                            df_now['nodeID'] == nodeID_prog_desc_unic[ii],
                            'nodeID_target'].values.astype(int)
                    if tarID:
                        _indx = np.where(
                                nodeID_prog_desc == nodeID_prog_desc_unic[ii])
                        obs_ref_local[_indx] = tarID
                df_prog['nodeID_target'] = pd.Series(obs_ref_local,
                                                     index=df_prog.index)

            snapcount += 1
        del nodeID_prog_desc
        del df_now, df_inter, df_prog
        return np.asarray(df_target['nodeID'].tolist()), progcounts



