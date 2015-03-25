""" Illustris Simulation: Public Data Release.
lhalotree.py: File I/O related to the LHaloTree merger tree files. """

import numpy as np
import h5py

from groupcat import gcPath
from util import partTypeNum

def treePath(basePath,chunkNum=0):
    """ Return absolute path to a LHaloTree HDF5 file (modify as needed). """
    filePath = basePath + '/trees/treedata/' + 'trees_sf1_135.' + str(chunkNum) + '.hdf5'
    
    return filePath
    
def treeOffsets(basePath, snapNum, id):
    """ Handle offset loading for a LHaloTree merger tree cutout. """
    # load groupcat chunk offsets from header of first file
    with h5py.File(gcPath(basePath,snapNum),'r') as f:
        groupFileOffsets = f['Header'].attrs['FileOffsets_Subhalo']
    
    # calculate target groups file chunk which contains this id
    groupFileOffsets = int(id) - groupFileOffsets
    fileNum = np.max( np.where(groupFileOffsets >= 0) )
    groupOffset = groupFileOffsets[fileNum]
    
    with h5py.File(gcPath(basePath,snapNum,fileNum),'r') as f:
        # load the merger tree offsets of this subgroup
        TreeFile  = f['Offsets']['Subhalo_LHaloTreeFile'][groupOffset]
        TreeIndex = f['Offsets']['Subhalo_LHaloTreeIndex'][groupOffset]
        TreeNum   = f['Offsets']['Subhalo_LHaloTreeNum'][groupOffset]
        return TreeFile,TreeIndex,TreeNum
    
def singleNodeFlat(conn, index, data_in, data_out, count, onlyMPB):
    """ Recursive helper function: Add a single tree node. """
    data_out[count] = data_in[index]
    
    count += 1
    count = recProgenitorFlat(conn,index,data_in,data_out,count,onlyMPB)
    
    return count
    
def recProgenitorFlat(conn, start_index, data_in, data_out, count, onlyMPB):
    """ Recursive helper function: Flatten out the unordered LHaloTree, one data field at a time. """
    firstProg = conn["FirstProgenitor"][start_index]
    
    if firstProg < 0:
        return count
        
    # depth-ordered traversal (down mpb)
    count = singleNodeFlat(conn,firstProg,data_in,data_out,count,onlyMPB)
    
    # explore breadth
    if not onlyMPB:
        nextProg = conn["NextProgenitor"][firstProg]
        
        while nextProg >= 0:
            count = singleNodeFlat(conn,nextProg,data_in,data_out,count,onlyMPB)
            
            nextProg = conn["NextProgenitor"][nextProg]
    firstProg = conn["FirstProgenitor"][firstProg]
        
    return count
    
def loadTree(basePath, snapNum, id, fields=None, onlyMPB=False):
    """ Load portion of LHaloTree, for a given subhalo, re-arranging into a flat format. """
    TreeFile,TreeIndex,TreeNum = treeOffsets(basePath, snapNum, id)
    
    # config
    gName  = 'Tree' + str(TreeNum) # group name containing this subhalo
    nRows  = None # we do not know in advance the size of the tree
    
    # make sure fields is not a single element
    if isinstance(fields, basestring):
        fields = [fields]
    
    fTree = h5py.File(treePath(basePath,TreeFile),'r')
    
    # if no fields requested, return everything
    if not fields:
        fields = fTree[gName].keys()
    
    # verify existence of requested fields
    for field in fields:
        if field not in fTree[gName].keys():
            raise Exception('Error: Requested field '+field+' not in tree.')
    
    # load connectivity for this entire TreeX group
    connFields = ['FirstProgenitor','NextProgenitor']
    conn = {}
    
    for field in connFields:
        conn[field] = fTree[gName][field][:]
            
    # determine sub-tree size with dummy walk
    dummy = np.zeros( conn['FirstProgenitor'].shape, dtype='int32' )
    nRows = singleNodeFlat(conn, TreeIndex, dummy, dummy, 0, onlyMPB)
          
    result = {}
    result['count'] = nRows
          
    # walk through connectivity, one data field at a time
    for field in fields:
        # load field for entire tree? doing so is much faster than randomly accessing the disk 
        # during walk, assuming that the sub-tree is a large fraction of the full tree, and that 
        # the sub-tree is large in the absolute sense. the decision is heuristic, and can be 
        # modified (if you have the tree on a fast SSD, could disable the full load).
        if nRows < 1000: # and float(nRows)/len(result['FirstProgenitor']) > 0.1
            # do not load, walk with single disk reads
            full_data = fTree[gName][field]
        else:
            # pre-load all, walk in-memory
            full_data = fTree[gName][field][:]
            
        # allocate the data array in the sub-tree
        dtype = fTree[gName][field].dtype
        shape = list(fTree[gName][field].shape)
        shape[0] = nRows
        
        data = np.zeros(shape, dtype=dtype)
        
        # walk the tree, depth-first
        count = singleNodeFlat(conn, TreeIndex, full_data, data, 0, onlyMPB)          
        
        # save field
        result[field] = data
    
    fTree.close()
    
    # only a single field? then return the array instead of a single item dict
    if len(fields) == 1:
        return result[fields[0]]
           
    return result
    
