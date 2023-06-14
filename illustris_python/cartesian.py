""" Illustris Simulation: Public Data Release.
cartesian.py: File I/O related to the cartesian output files (THESAN only). """
from __future__ import print_function

import numpy as np
import h5py
import six
from os.path import isfile


def cartPath(basePath, cartNum, chunkNum=0):
    """ Return absolute path to a cartesian HDF5 file (modify as needed). """
    filePath_list = [ f'{basePath}/cartesian_{cartNum:03d}/cartesian_{cartNum:03d}.{chunkNum}.hdf5',
                    ]

    for filePath in filePath_list:
        if isfile(filePath):
            return filePath

    raise ValueError("No cartesian file found!")

def getNumPixel(header):
    """ Calculate number of pixels (per dimension) given a cartesian header. """
    return header['NumPixels']

def loadSubset(basePath, cartNum, fields=None, bbox=None, sq=True):
    """ Load a subset of fields in the cartesian grids.
        If bbox is specified, load only that subset of data. bbox should have the 
           form [[start_i, start_j, start_k], [end_i, end_j, end_k]], where i,j,k are 
           the indices for x,y,z dimensions. Notice the last index is *inclusive*.
        If sq is True, return a numpy array instead of a dict if len(fields)==1. """
    result = {}

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(cartPath(basePath, cartNum), 'r') as f:
        header = dict(f['Header'].attrs.items())
        nPix = getNumPixel(header)

    # decide global read size, starting file chunk, and starting file chunk offset
    if bbox:
        load_all = False
        start_i, start_j, start_k = bbox[0]
        end_i, end_j, end_k = bbox[1]
        assert(start_i>=0)
        assert(start_j>=0)
        assert(start_k>=0)
        assert(end_i<nPix)
        assert(end_j<nPix)
        assert(end_k<nPix)
    else:
        load_all = True
        bbox = [[0, 0, 0], [nPix-1, nPix-1, nPix-1]]

    bbox = np.array(bbox)


    numToRead = (bbox[1,0]-bbox[0,0]+1) * (bbox[1,1]-bbox[0,1]+1) * (bbox[1,2]-bbox[0,2]+1)

    if numToRead==0:
        return result

    with h5py.File(cartPath(basePath, cartNum, 0), 'r') as f:
        # if fields not specified, load everything; otherwise check entry
        if not fields:
            fields = list(f.keys())
            fields.remove('Header')
        else:
            for field in fields:
                # verify existence
                if field not in f.keys():
                    raise Exception(f"Cartesian output does not have field [{field}]")

        for field in fields:
            # replace local length with global
            shape = list(f[field].shape)
            shape[0] = numToRead

            # allocate within return dict
            dtype = f[field].dtype
            result[field] = np.zeros(shape, dtype=dtype)

    # loop over chunks
    wOffset = 0
    fileOffset = 0
    origNumToRead = numToRead
    fileNum=0


    while numToRead:
        with h5py.File(cartPath(basePath, cartNum, fileNum), 'r') as f:

            # set local read length for this file chunk, truncate to be within the local size
            numPixelsLocal = f[fields[0]][()].shape[0]

            if load_all:
                pixToReadLocal = np.full(numPixelsLocal, True, dtype=bool)
                numToReadLocal = numPixelsLocal
            else:
                local_pixels_index = np.arange(fileOffset, fileOffset+numPixelsLocal)
                local_pixels_i = local_pixels_index//(nPix**2)
                local_pixels_j = (local_pixels_index-local_pixels_i*nPix**2)//nPix
                local_pixels_k = local_pixels_index-local_pixels_i*nPix**2-local_pixels_j*nPix

                pixToReadLocal = (local_pixels_i>=bbox[0,0]) & (local_pixels_i<=bbox[1,0]) &\
                                 (local_pixels_j>=bbox[0,1]) & (local_pixels_j<=bbox[1,1]) &\
                                 (local_pixels_k>=bbox[0,2]) & (local_pixels_k<=bbox[1,2])
                numToReadLocal = len(np.where(pixToReadLocal)[0])
                
            # loop over each requested field for this particle type
            for i, field in enumerate(fields):
                result[field][wOffset:wOffset+numToReadLocal] = f[field][pixToReadLocal]

            wOffset   += numToReadLocal
            numToRead -= numToReadLocal

            fileOffset += numPixelsLocal
            fileNum += 1

    # verify we read the correct number
    if origNumToRead != wOffset:
        raise Exception("Read ["+str(wOffset)+"] particles, but was expecting ["+str(origNumToRead)+"]")

    # only a single field? then return the array instead of a single item dict
    if sq and len(fields) == 1:
        return result[fields[0]]

    return result

