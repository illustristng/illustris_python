""" Illustris Simulation: Public Data Release.
sublink.py: File I/O related to the Sublink merger tree files. """

import numpy as np
import h5py
import glob
import six
import os

from .groupcat import gcPath, offsetPath
from .util import partTypeNum


def treePath(basePath, treeName, chunkNum=0):
    """ Return absolute path to a SubLink HDF5 file (modify as needed). """
    # tree_path = '/trees/' + treeName + '/' + 'tree_extended.' + str(chunkNum) + '.hdf5'
    tree_path = os.path.join('trees', treeName, 'tree_extended.' + str(chunkNum) + '.hdf5')

    _path = os.path.join(basePath, tree_path)
    if len(glob.glob(_path)):
        return _path

    # new path scheme
    _path = os.path.join(basePath, os.path.pardir, 'postprocessing', tree_path)
    if len(glob.glob(_path)):
        return _path

    # try one or more alternative path schemes before failing
    _path = os.path.join(basePath, 'postprocessing', tree_path)
    if len(glob.glob(_path)):
        return _path

    raise ValueError("Could not construct treePath from basePath = '{}'".format(basePath))


def treeOffsets(basePath, snapNum, id, treeName):
    """ Handle offset loading for a SubLink merger tree cutout. """
    # old or new format
    if 'fof_subhalo' in gcPath(basePath, snapNum) or treeName == "SubLink_gal":
        # load groupcat chunk offsets from separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['FileOffsets/Subhalo'][()]

        offsetFile = offsetPath(basePath, snapNum)
        prefix = 'Subhalo/' + treeName + '/'

        groupOffset = id
    else:
        # load groupcat chunk offsets from header of first file
        with h5py.File(gcPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['Header'].attrs['FileOffsets_Subhalo']

        # calculate target groups file chunk which contains this id
        groupFileOffsets = int(id) - groupFileOffsets
        fileNum = np.max(np.where(groupFileOffsets >= 0))
        groupOffset = groupFileOffsets[fileNum]

        offsetFile = gcPath(basePath, snapNum, fileNum)
        prefix = 'Offsets/Subhalo_Sublink'

    with h5py.File(offsetFile, 'r') as f:
        # load the merger tree offsets of this subgroup
        RowNum     = f[prefix+'RowNum'][groupOffset]
        LastProgID = f[prefix+'LastProgenitorID'][groupOffset]
        SubhaloID  = f[prefix+'SubhaloID'][groupOffset]
        return RowNum, LastProgID, SubhaloID

offsetCache = dict()

def subLinkOffsets(basePath, treeName, cache=True):
    # create quick offset table for rows in the SubLink files
    if cache is True:
        cache = offsetCache

    if type(cache) is dict:
        path = os.path.join(basePath, treeName)
        try:
            return cache[path]
        except KeyError:
            pass

    search_path = treePath(basePath, treeName, '*')
    numTreeFiles = len(glob.glob(search_path))
    if numTreeFiles == 0:
        raise ValueError("No tree files found! for path '{}'".format(search_path))
    offsets = np.zeros(numTreeFiles, dtype='int64')

    for i in range(numTreeFiles-1):
        with h5py.File(treePath(basePath, treeName, i), 'r') as f:
            offsets[i+1] = offsets[i] + f['SubhaloID'].shape[0]

    if type(cache) is dict:
        cache[path] = offsets

    return offsets

def loadTree(basePath, snapNum, id, fields=None, onlyMPB=False, onlyMDB=False, treeName="SubLink", cache=True):
    """ Load portion of Sublink tree, for a given subhalo, in its existing flat format.
        (optionally restricted to a subset fields)."""
    # the tree is all subhalos between SubhaloID and LastProgenitorID
    RowNum, LastProgID, SubhaloID = treeOffsets(basePath, snapNum, id, treeName)

    if RowNum == -1:
        print('Warning, empty return. Subhalo [%d] at snapNum [%d] not in tree.' % (id, snapNum))
        return None

    rowStart = RowNum
    rowEnd   = RowNum + (LastProgID - SubhaloID)
    nRows    = rowEnd - rowStart + 1

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    offsets = subLinkOffsets(basePath, treeName, cache)

    # find the tree file chunk containing this row
    rowOffsets = rowStart - offsets

    try:
        fileNum = np.max(np.where(rowOffsets >= 0))
    except ValueError as err:
        print("ERROR: ", err)
        print("rowStart = {}, offsets = {}, rowOffsets = {}".format(rowStart, offsets, rowOffsets))
        print(np.where(rowOffsets >= 0))
        raise
    fileOff = rowOffsets[fileNum]

    # load only main progenitor branch? in this case, get MainLeafProgenitorID now
    if onlyMPB:
        with h5py.File(treePath(basePath, treeName, fileNum), 'r') as f:
            MainLeafProgenitorID = f['MainLeafProgenitorID'][fileOff]

        # re-calculate nRows
        rowEnd = RowNum + (MainLeafProgenitorID - SubhaloID)
        nRows  = rowEnd - rowStart + 1

    # load only main descendant branch (e.g. from z=0 descendant to current subhalo)
    if onlyMDB:
        with h5py.File(treePath(basePath, treeName, fileNum),'r') as f:
            RootDescendantID = f['RootDescendantID'][fileOff]

        # re-calculate tree subset
        rowStart = RowNum - (SubhaloID - RootDescendantID) + 1
        rowEnd   = RowNum
        nRows    = rowEnd - rowStart + 1
        fileOff -= nRows

    # read
    result = {'count': nRows}

    with h5py.File(treePath(basePath, treeName, fileNum), 'r') as f:
        # if no fields requested, return all fields
        if not fields:
            fields = list(f.keys())

        if fileOff + nRows > f['SubfindID'].shape[0]:
            raise Exception('Should not occur. Each tree is contained within a single file.')

        # loop over each requested field
        for field in fields:
            if field not in f.keys():
                raise Exception("SubLink tree does not have field ["+field+"]")

            # read
            result[field] = f[field][fileOff:fileOff+nRows]

    # only a single field? then return the array instead of a single item dict
    if len(fields) == 1:
        return result[fields[0]]

    return result


def maxPastMass(tree, index, partType='stars'):
    """ Get maximum past mass (of the given partType) along the main branch of a subhalo
        specified by index within this tree. """
    ptNum = partTypeNum(partType)

    branchSize = tree['MainLeafProgenitorID'][index] - tree['SubhaloID'][index] + 1
    masses = tree['SubhaloMassType'][index: index + branchSize, ptNum]
    return np.max(masses)


def numMergers(tree, minMassRatio=1e-10, massPartType='stars', index=0):
    """ Calculate the number of mergers in this sub-tree (optionally above some mass ratio threshold). """
    # verify the input sub-tree has the required fields
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType']

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    numMergers   = 0
    invMassRatio = 1.0 / minMassRatio

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]
    fpID   = tree['FirstProgenitorID'][index]

    while fpID != -1:
        fpIndex = index + (fpID - rootID)
        fpMass  = maxPastMass(tree, fpIndex, massPartType)

        # explore breadth
        npID = tree['NextProgenitorID'][fpIndex]

        while npID != -1:
            npIndex = index + (npID - rootID)
            npMass  = maxPastMass(tree, npIndex, massPartType)

            # count if both masses are non-zero, and ratio exceeds threshold
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass

                if ratio >= minMassRatio and ratio <= invMassRatio:
                    numMergers += 1

            npID = tree['NextProgenitorID'][npIndex]

        fpID = tree['FirstProgenitorID'][fpIndex]

    return numMergers
