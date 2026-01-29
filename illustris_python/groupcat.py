""" Illustris Simulation: Public Data Release.
groupcat.py: File I/O related to the FoF and Subfind group catalogs. """
from __future__ import print_function

import six
from os import environ
from os.path import isfile,expanduser,join
import numpy as np
import h5py
from pathlib import Path

import multiprocessing as mp
from functools import partial


def gcPath(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a group catalog HDF5 file (modify as needed). """
    gcPath = basePath + '/groups_%03d/' % snapNum
    filePath1 = gcPath + 'groups_%03d.%d.hdf5' % (snapNum, chunkNum)
    filePath2 = gcPath + 'fof_subhalo_tab_%03d.%d.hdf5' % (snapNum, chunkNum)

    if isfile(expanduser(filePath1)):
        return filePath1
    return filePath2


def offsetPath(basePath, snapNum):
    """ Return absolute path to a separate offset file (modify as needed). """
    offsetPath = join(Path(basePath).parent, 'postprocessing/offsets/offsets_%03d.hdf5' % snapNum)

    return offsetPath


def _readfunc(basePath, snapNum, gName, nName, fields, i):
    """ Multiprocessing target for loadObjects() below. """
    result = {}

    f = h5py.File(gcPath(basePath, snapNum, i), 'r')

    if not f['Header'].attrs['N'+nName+'_ThisFile']:
        f.close()
        return None # empty file chunk

    # loop over each requested field and read data local to this chunk
    for field in fields:
        result[field] = f[gName][field][()]

    f.close()
    return result

def loadObjects(basePath, snapNum, gName, nName, fields, nThreads=None):
    """ Load either halo or subhalo information from the group catalog. """
    result = {}

    if nThreads is None:
        nThreads = environ.get('OMP_NUM_THREADS', 1)

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(gcPath(basePath, snapNum), 'r') as f:

        header = dict(f['Header'].attrs.items())

        if 'N'+nName+'_Total' not in header and nName == 'subgroups':
            nName = 'subhalos' # alternate convention

        result['count'] = np.int64(f['Header'].attrs['N' + nName + '_Total'])

        if not result['count']:
            print('warning: zero groups, empty return (snap=' + str(snapNum) + ').')
            return result

    # special case: single file? about x2 faster because of overhead of ndarray[:] = data, rather than ndarray = data.
    if header['NumFiles'] == 1:
        with h5py.File(gcPath(basePath, snapNum), 'r') as f:
            for field in fields:
                result[field] = f[gName][field][()]
        if len(fields) == 1:
            return result[fields[0]]
        return result

    with h5py.File(gcPath(basePath, snapNum), 'r') as f:
        # find a chunk with objects of this type
        i = 1
        while len(f[gName].keys()) == 0:
            f.close()
            f = h5py.File(gcPath(basePath, snapNum, i), 'r')
            i += 1

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for field in fields:
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Group catalog does not have requested field [" + field + "]!")

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = result['count']

            # allocate within return dict
            result[field] = np.zeros(shape, dtype=f[gName][field].dtype)

    # loop over chunks
    if mp.current_process().name != "MainProcess":
        nThreads = 1 # already inside daemonic child process, cannot spawn more children
        
    if nThreads == 1 or header['NumFiles'] == 1:
        # serial load
        wOffset = 0

        for i in range(header['NumFiles']):
            f = h5py.File(gcPath(basePath, snapNum, i), 'r')

            if not f['Header'].attrs['N'+nName+'_ThisFile']:
                f.close()
                continue  # empty file chunk

            # loop over each requested field
            for field in fields:
                if field not in f[gName].keys():
                    raise Exception("Group catalog does not have requested field [" + field + "]!")

                # shape and type
                shape = f[gName][field].shape

                # read data local to the current file
                result[field][wOffset:wOffset+shape[0], ...] = f[gName][field][()]

            wOffset += shape[0]
            f.close()
    else:
        # parallelized load
        pool = mp.Pool(processes=nThreads)

        fileNums = range(header['NumFiles'])
        func = partial(_readfunc,basePath,snapNum,gName,nName,fields)
        p_results = pool.map(func, fileNums)
        pool.close()

        # write
        wOffset = 0

        for i in range(header['NumFiles']):
            if p_results[i] is None:
                continue # no objects in this chunk

            for field in fields:
                numLoc = p_results[i][field].shape[0]
                result[field][wOffset:wOffset+numLoc,...] = p_results[i][field]

            wOffset += numLoc

    # only a single field? then return the array instead of a single item dict
    if len(fields) == 1:
        return result[fields[0]]

    return result


def loadSubhalos(basePath, snapNum, fields=None):
    """ Load all subhalo information from the entire group catalog for one snapshot
       (optionally restrict to a subset given by fields). """

    return loadObjects(basePath, snapNum, "Subhalo", "subgroups", fields)


def loadHalos(basePath, snapNum, fields=None):
    """ Load all halo information from the entire group catalog for one snapshot
       (optionally restrict to a subset given by fields). """

    return loadObjects(basePath, snapNum, "Group", "groups", fields)


def loadHeader(basePath, snapNum):
    """ Load the group catalog header. """
    with h5py.File(gcPath(basePath, snapNum), 'r') as f:
        header = dict(f['Header'].attrs.items())

    return header


def load(basePath, snapNum):
    """ Load complete group catalog all at once. """
    r = {}
    r['subhalos'] = loadSubhalos(basePath, snapNum)
    r['halos']    = loadHalos(basePath, snapNum)
    r['header']   = loadHeader(basePath, snapNum)
    return r


def loadSingle(basePath, snapNum, haloID=-1, subhaloID=-1):
    """ Return complete group catalog information for one halo or subhalo. """
    if (haloID < 0 and subhaloID < 0) or (haloID >= 0 and subhaloID >= 0):
        raise Exception("Must specify either haloID or subhaloID (and not both).")

    gName = "Subhalo" if subhaloID >= 0 else "Group"
    searchID = subhaloID if subhaloID >= 0 else haloID

    # old or new format
    if 'fof_subhalo' in gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            offsets = f['FileOffsets/'+gName][()]
    else:
        # use header of group catalog
        with h5py.File(gcPath(basePath, snapNum), 'r') as f:
            offsets = f['Header'].attrs['FileOffsets_'+gName]

    offsets = searchID - offsets
    fileNum = np.max(np.where(offsets >= 0))
    groupOffset = offsets[fileNum]

    # load halo/subhalo fields into a dict
    result = {}

    with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
        for haloProp in f[gName].keys():
            result[haloProp] = f[gName][haloProp][groupOffset]

    return result
