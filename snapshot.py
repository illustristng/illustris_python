""" Illustris Simulation: Public Data Release.
snapshot.py: File I/O related to the snapshot files. """

import numpy as np
import h5py
from util import partTypeNum
from groupcat import gcPath

def snapPath(basePath,snapNum,chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    snapPath = basePath + '/snapdir_' + str(snapNum).zfill(3) + '/'
    filePath = snapPath + 'snap_' + str(snapNum).zfill(3)
    filePath += '.' + str(chunkNum) + '.hdf5'
    
    return filePath

def getNumPart(header):
    """ Calculate number of particles of all types given a snapshot header. """
    nTypes = 6
    
    nPart = np.zeros( nTypes, dtype=np.int64 )
    for j in range(nTypes):
        nPart[j] = header['NumPart_Total'][j] | (header['NumPart_Total_HighWord'][j] << 32)
        
    return nPart    
    
def loadSubset(basePath,snapNum,partType,fields=None,subset=None,mdi=None):
    """ Load a subset of fields for all particles/cells of a given partType.
        If offset and length specified, load only that subset of the partType. """
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
            
        # decide global read size, starting file chunk, and starting file chunk offset
        if subset:
            offsetsThisType = subset['offsetType'][ptNum] - subset['snapOffsets'][ptNum,:]
            
            fileNum = np.max(np.where( offsetsThisType >= 0 ))
            fileOff = offsetsThisType[fileNum]
            numToRead = subset['lenType'][ptNum]
        else:
            fileNum = 0
            fileOff = 0
            numToRead = nPart[ptNum]
            
        result['count'] = numToRead
        
        if not numToRead:
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
        
        for i, field in enumerate(fields):                
            # verify existence
            if not field in f[gName].keys():
                raise Exception("Particle type ["+str(ptNum)+"] does not have field ["+field+"]")
                
            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = numToRead
            
            # multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception("Read error: mdi requested on non-2D field ["+field+"]")
                shape = [shape[0]]

            #print 'Load:',partType,field,numToRead,'of',nPart[ptNum],f[gName][field].dtype,'shape',shape
        
            # allocate within return dict
            result[field] = np.zeros( shape, dtype=f[gName][field].dtype )
        
    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead
    
    while numToRead:
        f = h5py.File(snapPath(basePath,snapNum,fileNum),'r')
        
        # no particles of requested type in this file chunk?
        if not gName in f:
            if subset:
                raise Exception('Read error: subset read should be contiguous.')
            fileNum += 1
            continue
        
        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]
        
        numToReadLocal = numToRead
        
        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = numTypeLocal - fileOff
        
        #print '['+str(fileNum).rjust(3)+'] off='+str(fileOff)+' read ['+str(numToReadLocal)+\
        #      '] of ['+str(numTypeLocal)+'] remaining = '+str(numToRead-numToReadLocal)
        
        # loop over each requested field for this particle type
        for i, field in enumerate(fields):
            # read data local to the current file
            if mdi is None or mdi[i] is None:
                result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            else:
                result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal,mdi[i]]
        
        wOffset   += numToReadLocal
        numToRead -= numToReadLocal
        fileNum   += 1
        fileOff    = 0 # start at beginning of all file chunks other than the first
        
        f.close()
        
    # verify we read the correct number
    if origNumToRead != wOffset:
        raise Exception("Read ["+str(wOffset)+"] particles, but was expecting ["+str(origNumToRead)+"]")
        
    # only a single field? then return the array instead of a single item dict
    if len(fields) == 1:
        return result[fields[0]]
        
    return result
        
def getSnapOffsets(basePath,snapNum,id,type):
    """ Compute offsets within snapshot for a particular group/subgroup. """
    r = {}
    
    # load groupcat chunk offsets from header of first file
    with h5py.File(gcPath(basePath,snapNum),'r') as f:
        groupFileOffsets = f['Header'].attrs['FileOffsets_'+type]

        # load the offsets for the snapshot chunks
        r['snapOffsets'] = f['Header'].attrs['FileOffsets_Snap']
    
    # calculate target groups file chunk which contains this id
    groupFileOffsets = int(id) - groupFileOffsets
    fileNum = np.max( np.where(groupFileOffsets >= 0) )
    groupOffset = groupFileOffsets[fileNum]
    
    # load the length (by type) of this group/subgroup from the group catalog
    with h5py.File(gcPath(basePath,snapNum,fileNum),'r') as f:
        r['lenType'] = f[type][type+'LenType'][groupOffset,:]
        
        # load the offset (by type) of this group/subgroup within the snapshot
        r['offsetType'] = f['Offsets'][type+'_SnapByType'][groupOffset,:]
    
    return r
        
def loadSubhalo(basePath,snapNum,id,partType,fields=None):
    """ Load all particles/cells of one type for a specific subhalo
        (optionally restricted to a subset fields). """
    # load subhalo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath,snapNum,id,"Subhalo")
    return loadSubset(basePath,snapNum,partType,fields,subset=subset)
    
def loadHalo(basePath,snapNum,id,partType,fields=None):
    """ Load all particles/cells of one type for a specific halo
        (optionally restricted to a subset fields). """
    # load halo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath,snapNum,id,"Group")
    return loadSubset(basePath,snapNum,partType,fields,subset=subset)
