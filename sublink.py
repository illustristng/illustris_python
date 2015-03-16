""" Illustris Simulation: Public Data Release.
sublink.py: File I/O related to the Sublink merger tree files. """

import numpy as np
import h5py
import pdb
import glob

from groupcat import gcPath

def treePath(basePath,chunkNum=0):
    """ Return absolute path to a SubLink HDF5 file (modify as needed). """
    filePath = basePath + '/trees/SubLink/' + 'tree_extended.' + str(chunkNum) + '.hdf5'
    
    return filePath
    
def treeOffsets(basePath, snapNum, id):
    """ Handle offset loading for a SubLink merger tree cutout. """
    # load groupcat chunk offsets from header of first file
    with h5py.File(gcPath(basePath,snapNum),'r') as f:
        groupFileOffsets = f['Header'].attrs['FileOffsets_Subhalo']
    
    # calculate target groups file chunk which contains this id
    groupFileOffsets = int(id) - groupFileOffsets
    fileNum = np.max( np.where(groupFileOffsets >= 0) )
    groupOffset = groupFileOffsets[fileNum]
    
    with h5py.File(gcPath(basePath,snapNum,fileNum),'r') as f:
        # load the merger tree offsets of this subgroup
        RowNum     = f['Offsets']['Subhalo_SublinkRowNum'][groupOffset]
        LastProgID = f['Offsets']['Subhalo_SublinkLastProgenitorID'][groupOffset]
        SubhaloID  = f['Offsets']['Subhalo_SublinkSubhaloID'][groupOffset]
        return RowNum,LastProgID,SubhaloID
        
def loadTree(basePath, snapNum, id, fields=None):
    """ Load portion of Sublink tree, for a given subhalo, in its existing flat format. """    
    # the tree is all subhalos between SubhaloID and LastProgenitorID
    RowNum,LastProgID,SubhaloID = treeOffsets(basePath, snapNum, id)
    
    rowStart = RowNum
    rowEnd   = RowNum + (LastProgID - SubhaloID)
    nRows    = rowEnd - rowStart + 1
    
    # create quick offset table for rows in the SubLink files
    numTreeFiles = len(glob.glob(treePath(basePath,'*')))
    offsets = np.zeros( numTreeFiles, dtype='int64' )
    
    for i in range(numTreeFiles-1):
        with h5py.File(treePath(basePath,i),'r') as f:
            offsets[i+1] = offsets[i] + f['SubhaloID'].shape[0]
        
    # find the tree file chunk containing this row
    rowOffsets = rowStart - offsets
    fileNum = np.max(np.where( rowOffsets >= 0 ))
    fileOff = rowOffsets[fileNum]
    
    # loop over chunks
    wOffset = 0
    nRowsOrig = nRows
    
    result = {}
    
    while nRows:
        f = h5py.File(treePath(basePath,fileNum),'r')
        
        # if no fields requested, return everything
        if not fields:
            fields = f.keys()
        
        # set local read length for this file chunk, truncate to be within the local size
        nRowsLocal = nRows
        
        if fileOff + nRowsLocal > f['SubfindID'].shape[0]:
            nRowsLocal = f['SubfindID'].shape[0] - fileOff
        
        #print '['+str(fileNum).rjust(3)+'] off='+str(fileOff)+' read ['+str(nRowsLocal)+\
        #      '] of ['+str(f['SubfindID'].shape[0])+'] remaining = '+str(nRows-nRowsLocal)
        
        # loop over each requested field: allocate if not already
        for field in fields:
            if field not in f.keys():
                raise Exception("SubLink tree does not have field ["+field+"]")
                
            if field not in result:
                shape = list(f[field].shape)
                shape[0] = nRows
                
                result[field] = np.zeros( tuple(shape), dtype=f[field].dtype )
                
        # loop over each requested field: read and save
        for field in fields:
            result[field][wOffset:wOffset+nRowsLocal] = f[field][fileOff:fileOff+nRowsLocal]
            
        wOffset += nRowsLocal
        nRows   -= nRowsLocal
        fileNum += 1
        fileOff  = 0
            
        f.close()
            
    if nRowsOrig != wOffset:
        raise Exception("Read ["+str(wOffset)+"] entries, but was expecting ["+str(nRowsOrig)+"]!")
           
    # only a single field? then return the array instead of a single item dict
    if len(fields) == 1:
        return result[fields[0]]
           
    return result
