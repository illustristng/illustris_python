""" Illustris Simulation: Public Data Release.
snapshot.py: File I/O related to the snapshot files. """
from __future__ import print_function

import numpy as np
import h5py
import six
from os.path import isfile
import astropy.units as u

from .util import partTypeNum
from .groupcat import gcPath, offsetPath, loadSingle

code_mass = u.def_unit('code_mass', 1.0e10 * u.solMass)
code_length = u.def_unit('code_length', u.kpc)
code_velocity = u.def_unit('code_velocity', u.km / u.s)

def snapPath(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    snapPath = basePath + '/snapdir_' + str(snapNum).zfill(3) + '/'
    filePath1 = snapPath + 'snap_' + str(snapNum).zfill(3) + '.' + str(chunkNum) + '.hdf5'
    filePath2 = filePath1.replace('/snap_', '/snapshot_')

    if isfile(filePath1):
        return filePath1
    return filePath2

def getNumPart(header):
    """ Calculate number of particles of all types given a snapshot header. """
    if 'NumPart_Total_HighWord' not in header:
        return np.int64(header['NumPart_Total']) # new uint64 convention

    nTypes = 6

    nPart = np.zeros(nTypes, dtype=np.int64)
    for j in range(nTypes):
        nPart[j] = header['NumPart_Total'][j] | (np.int64(header['NumPart_Total_HighWord'][j]) << 32)

    return nPart


def loadSubset(basePath, snapNum, partType, fields=None, subset=None, mdi=None, sq=True, float32=False):
    """ Load a subset of fields for all particles/cells of a given partType.
        If offset and length specified, load only that subset of the partType.
        If mdi is specified, must be a list of integers of the same length as fields,
        giving for each field the multi-dimensional index (on the second dimension) to load.
          For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
          of y-Coordinates only, together with Masses.
        If sq is True, return a numpy array instead of a dict if len(fields)==1.
        If float32 is True, load any float64 datatype arrays directly as float32 (save memory). """
    result = {}

    ptNum = partTypeNum(partType)
    gName = "PartType" + str(ptNum)

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(snapPath(basePath, snapNum), 'r') as f:

        header = dict(f['Header'].attrs.items())
        nPart = getNumPart(header)

        # decide global read size, starting file chunk, and starting file chunk offset
        if subset:
            offsetsThisType = subset['offsetType'][ptNum] - subset['snapOffsets'][ptNum, :]

            fileNum = np.max(np.where(offsetsThisType >= 0))
            fileOff = offsetsThisType[fileNum]
            numToRead = subset['lenType'][ptNum]
        else:
            fileNum = 0
            fileOff = 0
            numToRead = nPart[ptNum]

        result['count'] = numToRead

        if not numToRead:
            # print('warning: no particles of requested type, empty return.')
            return result

        # find a chunk with this particle type
        i = 1
        while gName not in f:
            f = h5py.File(snapPath(basePath, snapNum, i), 'r')
            i += 1

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for i, field in enumerate(fields):
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Particle type ["+str(ptNum)+"] does not have field ["+field+"]")

            # replace local length with global
            shape = list(f[gName][field].shape)
            dset_attrs = dict(f[gName][field].attrs.items())
            shape[0] = numToRead

            # multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception("Read error: mdi requested on non-2D field ["+field+"]")
                shape = [shape[0]]

            # allocate within return dict
            dtype = f[gName][field].dtype
            if dtype == np.float64 and float32: dtype = np.float32
            try:
                result[field] = np.zeros(shape, dtype=dtype) * (code_mass**(dset_attrs['mass_scaling']) * 
                                                                code_length**(dset_attrs['length_scaling']) * 
                                                                code_velocity**(dset_attrs['velocity_scaling']))
            except KeyError:
                result[field] = np.zeros(shape, dtype=dtype)


    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead

    while numToRead:
        f = h5py.File(snapPath(basePath, snapNum, fileNum), 'r')
        Header = dict(f['Header'].attrs.items())

        # no particles of requested type in this file chunk?
        if gName not in f:
            f.close()
            fileNum += 1
            fileOff  = 0
            continue

        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]

        numToReadLocal = numToRead

        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = int(numTypeLocal - fileOff)

        #print('['+str(fileNum).rjust(3)+'] off='+str(fileOff)+' read ['+str(numToReadLocal)+\
        #      '] of ['+str(numTypeLocal)+'] remaining = '+str(numToRead-numToReadLocal))

        # loop over each requested field for this particle type
        for i, field in enumerate(fields):
            # read data local to the current file
            dset = f[gName][field]
            dset_attrs = dict(dset.attrs.items())
            if mdi is None or mdi[i] is None:
                try:
                    result[field][wOffset:wOffset+numToReadLocal] = (dset[fileOff:fileOff+numToReadLocal] * Header['Time']**(dset_attrs['a_scaling']) * 
                                                                     Header['HubbleParam']**(dset_attrs['h_scaling']) * code_mass**(dset_attrs['mass_scaling']) * 
                                                                     code_length**(dset_attrs['length_scaling']) * code_velocity**(dset_attrs['velocity_scaling']))
                except KeyError:
                    result[field][wOffset:wOffset+numToReadLocal] = dset[fileOff:fileOff+numToReadLocal]
            else:
                try:
                    result[field][wOffset:wOffset+numToReadLocal] = (dset[fileOff:fileOff+numToReadLocal, mdi[i]] * Header['Time']**(dset_attrs['a_scaling']) * 
                                                                     Header['HubbleParam']**(dset_attrs['h_scaling']) * code_mass**(dset_attrs['mass_scaling']) * 
                                                                     code_length**(dset_attrs['length_scaling']) * code_velocity**(dset_attrs['velocity_scaling']))
                except KeyError:
                    result[field][wOffset:wOffset+numToReadLocal] = dset[fileOff:fileOff+numToReadLocal, mdi[i]]

        wOffset   += numToReadLocal
        numToRead -= numToReadLocal
        fileNum   += 1
        fileOff    = 0  # start at beginning of all file chunks other than the first

        f.close()

    # verify we read the correct number
    if origNumToRead != wOffset:
        raise Exception("Read ["+str(wOffset)+"] particles, but was expecting ["+str(origNumToRead)+"]")

    # only a single field? then return the array instead of a single item dict
    if sq and len(fields) == 1:
        return result[fields[0]]

    return result


def getSnapOffsets(basePath, snapNum, id, type):
    """ Compute offsets within snapshot for a particular group/subgroup. """
    r = {}

    # old or new format
    if 'fof_subhalo' in gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['FileOffsets/'+type][()]
            r['snapOffsets'] = np.transpose(f['FileOffsets/SnapByType'][()])  # consistency
    else:
        # load groupcat chunk offsets from header of first file
        with h5py.File(gcPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['Header'].attrs['FileOffsets_'+type]
            r['snapOffsets'] = f['Header'].attrs['FileOffsets_Snap']

    # calculate target groups file chunk which contains this id
    groupFileOffsets = int(id) - groupFileOffsets
    fileNum = np.max(np.where(groupFileOffsets >= 0))
    groupOffset = groupFileOffsets[fileNum]

    # load the length (by type) of this group/subgroup from the group catalog
    with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
        r['lenType'] = f[type][type+'LenType'][groupOffset, :]

    # old or new format: load the offset (by type) of this group/subgroup within the snapshot
    if 'fof_subhalo' in gcPath(basePath, snapNum):
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            r['offsetType'] = f[type+'/SnapByType'][id, :]

            # add TNG-Cluster specific offsets if present
            if 'OriginalZooms' in f:
                for key in f['OriginalZooms']:
                    r[key] = f['OriginalZooms'][key][()] 
    else:
        with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
            r['offsetType'] = f['Offsets'][type+'_SnapByType'][groupOffset, :]

    return r


def loadSubhalo(basePath, snapNum, id, partType, fields=None):
    """ Load all particles/cells of one type for a specific subhalo
        (optionally restricted to a subset fields). """
    # load subhalo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath, snapNum, id, "Subhalo")
    return loadSubset(basePath, snapNum, partType, fields, subset=subset)


def loadHalo(basePath, snapNum, id, partType, fields=None):
    """ Load all particles/cells of one type for a specific halo
        (optionally restricted to a subset fields). """
    # load halo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath, snapNum, id, "Group")
    return loadSubset(basePath, snapNum, partType, fields, subset=subset)


def loadOriginalZoom(basePath, snapNum, id, partType, fields=None):
    """ Load all particles/cells of one type corresponding to an
        original (entire) zoom simulation. TNG-Cluster specific.
        (optionally restricted to a subset fields). """
    # load fuzz length, compute offset, call loadSubset                                                                     
    subset = getSnapOffsets(basePath, snapNum, id, "Group")

    # identify original halo ID and corresponding index
    halo = loadSingle(basePath, snapNum, haloID=id)
    assert 'GroupOrigHaloID' in halo, 'Error: loadOriginalZoom() only for the TNG-Cluster simulation.'
    orig_index = np.where(subset['HaloIDs'] == halo['GroupOrigHaloID'])[0][0]

    # (1) load all FoF particles/cells
    subset['lenType'] = subset['GroupsTotalLengthByType'][orig_index, :]
    subset['offsetType'] = subset['GroupsSnapOffsetByType'][orig_index, :]

    data1 = loadSubset(basePath, snapNum, partType, fields, subset=subset)

    # (2) load all non-FoF particles/cells
    subset['lenType'] = subset['OuterFuzzTotalLengthByType'][orig_index, :]
    subset['offsetType'] = subset['OuterFuzzSnapOffsetByType'][orig_index, :]

    data2 = loadSubset(basePath, snapNum, partType, fields, subset=subset)

    # combine and return
    if isinstance(data1, np.ndarray):
        # protect against empty data
        if isinstance(data2, dict):
            return data1
        return np.concatenate((data1,data2), axis=0)
    
    # protect against empty data
    if data1["count"] == 0:
        return data2
    elif data2["count"] == 0:
        return data1
    
    data = {'count':data1['count']+data2['count']}
    for key in data1.keys():
        if key == 'count': continue
        data[key] = np.concatenate((data1[key],data2[key]), axis=0)
    return data

