""" Illustris Simulation: Public Data Release.
groupcat.py: File I/O related to the FoF and Subfind group catalogs. """

import numpy as np
import h5py

def gcPath(basePath,snapNum,chunkNum=0):
    """ Return absolute path to a group catalog HDF5 file (modify as needed). """
    gcPath = basePath + '/groups_' + str(snapNum).zfill(3) + '/'
    filePath = gcPath + 'fof_subhalo_tab_' + str(snapNum).zfill(3)
    filePath += '.' + str(chunkNum) + '.hdf5'
    
    return filePath

def loadObjects(basePath,snapNum,gName,nName,fields):
    """ Load either halo or subhalo information from the group catalog. """
    result = {}
    
    # make sure fields is not a single element
    if isinstance(fields, basestring):
        fields = [fields]    
    
    # load header from first chunk
    with h5py.File(gcPath(basePath,snapNum),'r') as f:

        header = dict( f['Header'].attrs.items() )
        result['count'] = f['Header'].attrs['N'+nName+'_Total']
        
        if not result['count']:
            print 'warning: zero groups, empty return (snap='+str(snapNum)+').'
            return result
        
        # if fields not specified, load everything
        if not fields:
            fields = f[gName].keys()
        
        for field in fields:
            # verify existence
            if not field in f[gName].keys():
                raise Exception("Group catalog does not have requested field ["+field+"]!")
                
            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = result['count']
        
            # allocate within return dict
            result[field] = np.zeros( shape, dtype=f[gName][field].dtype )
        
    # loop over chunks
    wOffset = 0
    
    for i in range(header['NumFiles']):
        filePath = gcPath(basePath,snapNum,i)
        f = h5py.File(filePath,'r')
        
        if not f['Header'].attrs['N'+nName+'_ThisFile']:
            continue # empty file chunk
        
        # loop over each requested field for this particle type
        for field in fields:
            # shape and type
            shape = f[gName][field].shape
 
            # read data local to the current file (allocate dataset if it does not already exist)
            if len(shape) == 1:
                result[field][wOffset:wOffset+shape[0]] = f[gName][field][0:shape[0]]
            else:
                result[field][wOffset:wOffset+shape[0],:] = f[gName][field][0:shape[0],:]
        

        wOffset += shape[0]
        f.close()
        
    return result
        
def loadSubhalos(basePath,snapNum,fields=None):
    """ Load all subhalo information from the entire group catalog for one snapshot
       (optionally restrict to a subset given by fields). """
    
    return loadObjects(basePath,snapNum,"Subhalo","subgroups",fields)

def loadHalos(basePath,snapNum,fields=None):
    """ Load all halo information from the entire group catalog for one snapshot
       (optionally restrict to a subset given by fields). """
    
    return loadObjects(basePath,snapNum,"Group","groups",fields)
    
def loadHeader(basePath,snapNum):
    """ Load the group catalog header. """
    with h5py.File(gcPath(basePath,snapNum),'r') as f:
        header = dict( f['Header'].attrs.items() )
        
    return header
    
def load(basePath,snapNum):
    """ Load complete group catalog at once. """
    r = {}
    r['subhalos'] = loadSubhalos(basePath,snapNum)
    r['halos']    = loadHalos(basePath,snapNum)
    r['header']   = loadHeader(basePath,snapNum)
    return r
    
def loadSingle(fileBase,snapNum,haloID=None,subhaloID=None):
    """ Return complete subhalo (subfind) information on one group. 
        Search for subhalo/subgroup if getGroup==0, else search for halo/fof group. """

    if (not haloID and not subhaloID) or (haloID and subhaloID):
        raise Exception("Must specify either haloID or subhaloID (and not both).")
        
    dsetName = "Subhalo" if subhaloID else "Group"
 
    # load groupcat offsets, calculate target file and offset
    filePath = fileBase + '/postprocessing/offsets/offsets_'+dsetName.lower()+'_'+str(snapNum)+'.npy'
    offsets = np.load(filePath)
 
    offsets = searchID - offsets
    fileNum = np.max( np.where(offsets >= 0) )
    groupOffset = offsets[fileNum]
 
    # load subhalo fields into a dict
    filePath = fileBase + '/output/groups_' + str(snapNum).zfill(3) + '/'
    filePath += 'fof_subhalo_tab_' + str(snapNum).zfill(3) + '.' + str(fileNum) + '.hdf5'
 
    result = {}
 
    f = h5py.File(filePath,'r')
 
    for haloProp in f[dsetName].keys():
        result[haloProp] = f[dsetName][haloProp][groupOffset]
 
    f.close()
    return result
    