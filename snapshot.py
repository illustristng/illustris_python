""" Illustris Simulation: Public Data Release.
snapshot.py: File I/O related to the snapshot files. """

import numpy as np
import h5py
import pdb
from util import partTypeNum

def snapPath(basePath,snapNum,chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    snapPath = basePath + '/snapdir_' + str(snapNum).zfill(3) + '/'
    filePath = snapPath + 'snap_' + str(snapNum).zfill(3)
    filePath += '.' + str(chunkNum) + '.hdf5'
    
    return filePath

def getNumPart(header):
    """ Calculate number of particles of all types given a snapshot header. """
    nTypes = 6
    
    nPart = np.zeros( 6, dtype=np.int64 )
    for j in range(6):
        nPart[j] = header['NumPart_Total'][j] | (header['NumPart_Total_HighWord'][j] << 32)
        
    return nPart    
    
def loadSubset(basePath,snapNum,partType,fields):
    """ Load a subset of fields for all particles/cells of a given partType. """
    result = {}
    
    ptNum = partTypeNum(partType)
    gName = "PartType" + str(ptNum)
    
    # make sure fields is not a single element
    if isinstance(fields, basestring):
        fields = [fields]
    
    # load header from first chunk
    with h5py.File(snapPath(basePath,snapNum),'r') as f:

        header = dict( f['Header'].attrs.items() )
        nPart = getNumPart(header)
        result['count'] = nPart[ptNum]
        
        if not nPart[ptNum]:
            print 'warning: no particles of requested type, empty return.'
            return result
            
        # find a chunk with this particle type
        i = 1        
        while gName not in f:
            f = h5py.File(snapPath(basePath,snapNum,i),'r')
            i += 1
            
        # if fields not specified, load everything
        if not fields:
            fields = f[gName].keys()
        
        for field in fields:                
            # verify existence
            if not field in f[gName].keys():
                raise Exception("Particle type ["+str(ptNum)+"] does not have field ["+field+"]!")
                
            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = nPart[ptNum]
            
            #print 'Load:',partType,field,nPart[ptNum],f[gName][field].dtype
        
            # allocate within return dict
            result[field] = np.zeros( shape, dtype=f[gName][field].dtype )
        
    # loop over chunks
    wOffset = 0
    
    for i in range(header['NumFilesPerSnapshot']):
        filePath = snapPath(basePath,snapNum,i)
        f = h5py.File(filePath,'r')
        
        if not gName in f:
            continue # no particles of requested type in this file chunk
        
        # loop over each requested field for this particle type
        for field in fields:
            # read data local to the current file, stamp
            shape = f[gName][field].shape
            
            if len(shape) == 1:
                result[field][wOffset:wOffset+shape[0]] = f[gName][field][0:shape[0]]
            else:
                result[field][wOffset:wOffset+shape[0],:] = f[gName][field][0:shape[0],:]
        

        wOffset += shape[0]
        f.close()
        
    # verify we read the correct number
    if nPart[ptNum] != wOffset:
        raise Exception("Read ["+str(wOffset)+"] particles, but was expecting ["+str(nPart[ptNum])+"]!")
        
    return result
        
def loadSubhalo():
    # compute offsets, call loadSubset
    pass
    
def loadHalo():
    # compute offset, call loadSubset
    pass
