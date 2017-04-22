"""
"""
from nose.tools import assert_true, assert_false, assert_equal, assert_raises

import sys
import os
import nose
import numpy as np

BASE_PATH = "/n/hernquistfs1/Illustris/Runs/L75n1820FP/"

# Add path to directory containing 'illustris_python' module
#    e.g. if this file is in '/n/home00/lkelley/illustris/illustris_python/tests/'
this_path = os.path.realpath(__file__)
ill_py_path = os.path.abspath(os.path.join(this_path, os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(ill_py_path)
import illustris_python as ill


def test_groupcat_loadHalos():
    fields = ['GroupFirstSub']
    snap = 135
    group_first_sub = ill.groupcat.loadHalos(BASE_PATH, snap, fields=fields)
    print("group_first_sub.shape = ", group_first_sub.shape)
    assert_true(group_first_sub.shape == (7713601,))
    print("group_first_sub = ", group_first_sub)
    assert_true(np.all(group_first_sub[:3] == [0, 16937, 30430]))

    return


def test_groupcat_loadSubhalos():
    fields = ['SubhaloMass', 'SubhaloSFRinRad']
    snap = 135
    subhalos = ill.groupcat.loadSubhalos(BASE_PATH, snap, fields=fields)
    print("subhalos['SubhaloMass'] = ", subhalos['SubhaloMass'].shape)
    assert_true(subhalos['SubhaloMass'].shape == (4366546,))
    print("subhalos['SubhaloMass'] = ", subhalos['SubhaloMass'])
    assert_true(
        np.allclose(subhalos['SubhaloMass'][:3], [2.21748203e+04, 2.21866333e+03, 5.73408325e+02]))

    return


def test_snapshot_partTypeNum():
    names = ['gas', 'dm', 'stars', 'blackhole']
    nums = [0, 1, 4, 5]

    for name, num in zip(names, nums):
        pn = ill.snapshot.partTypeNum(name)
        print("\npartTypeNum('{}') = '{}' (should be '{}')".format(name, pn, num))
        assert_equal(pn, num)

    return


if __name__ == "__main__":
    module_name = sys.modules[__name__].__file__
    # result = nose.run(argv=[sys.argv[0], module_name, '-v', '--nocapture'])
    print("sys.argv = ", sys.argv)
    nose_args = [sys.argv[0], module_name]
    if len(sys.argv) > 1:
        nose_args.extend(sys.argv[1:])
    result = nose.run(argv=nose_args)
