"""Tests for the `illustris_python.sublink` submodule.

Running Tests
-------------
To run all tests, this script can be executed as:
    `$ python tests/sublink_test.py [-v] [--nocapture]`
from the root directory.

Alternatively, `nosetests` can be run and it will find the tests:
    `$ nosetests [-v] [--nocapture]`

To run particular tests (for example),
    `$ nosetests tests/sublink_test.py:test_loadTree`

To include coverage information,
    `$ nosetests --with-coverage --cover-package=.`

"""
import os
import glob
import numpy as np
from nose.tools import assert_equal, assert_raises, assert_true, assert_false

# `illustris_python` is imported as `ill` in local `__init__.py`
from . import ill, BASE_PATH_ILLUSTRIS_1


def test_treePath_1():
    tree_name = "SubLink"
    _path = ill.sublink.treePath(BASE_PATH_ILLUSTRIS_1, tree_name, '*')
    paths = glob.glob(_path)
    assert_false(len(paths) == 0)
    assert_true(os.path.exists(paths[0]))
    return


def test_treePath_2():
    # Construct a path that should fail
    tree_name = "SubLinkFail"
    assert_raises(ValueError, ill.sublink.treePath, BASE_PATH_ILLUSTRIS_1, tree_name, '*')
    return


def test_loadTree():
    fields = ['SubhaloMass', 'SubfindID', 'SnapNum']
    snap = 135
    start = 100

    # Values for Illustris-1, snap=135, start=100
    snap_num_last = [22, 25, 22, 21, 22]
    subhalo_mass_last = [0.0104475, 0.0105625, 0.0133463, 0.0138612, 0.0117906]

    group_first_sub = ill.groupcat.loadHalos(BASE_PATH_ILLUSTRIS_1, snap, fields=['GroupFirstSub'])

    for ii, nn, mm in zip(range(start, start+5), snap_num_last, subhalo_mass_last):
        tree = ill.sublink.loadTree(
            BASE_PATH_ILLUSTRIS_1, 135, group_first_sub[ii], fields=fields, onlyMPB=True)
        assert_equal(tree['SnapNum'][-1], nn)
        assert_true(np.isclose(tree['SubhaloMass'][-1], mm))

    return


def test_numMergers():
    snap = 135
    ratio = 1.0/5.0
    start = 100

    # Values for Illustris-1, snap=135, start=100
    num_mergers = [2, 2, 3, 4, 3]

    group_first_sub = ill.groupcat.loadHalos(BASE_PATH_ILLUSTRIS_1, snap, fields=['GroupFirstSub'])

    # the following fields are required for the walk and the mass ratio analysis
    fields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
              'FirstProgenitorID', 'SubhaloMassType']
    for i, nm in zip(range(start, start+5), num_mergers):
        tree = ill.sublink.loadTree(BASE_PATH_ILLUSTRIS_1, snap, group_first_sub[i], fields=fields)
        _num_merg = ill.sublink.numMergers(tree, minMassRatio=ratio)
        print("group_first_sub[{}] = {}, num_mergers = {} (should be {})".format(
            i, group_first_sub[i], _num_merg, nm))
        assert_equal(_num_merg, nm)

    return
