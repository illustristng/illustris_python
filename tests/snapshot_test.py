"""Tests for the `illustris_python` module.

Running Tests
-------------
To run all tests, this script can be executed as:
    `$ python tests/test.py [-v] [--nocapture]`
from the root directory.

Alternatively, `nosetests` can be run and it will find the tests:
    `$ nosetests [-v] [--nocapture]`

To run particular tests (for example),
    `$ nosetests tests/test.py:test_groupcat_loadSubhalos`

"""
from nose.tools import assert_true, assert_equal, assert_raises

# `illustris_python` is imported as `ill` in local `__init__.py`
from . import ill, BASE_PATH_ILLUSTRIS_1


def test_snapshot_partTypeNum_1():
    names = ['gas', 'dm', 'tracers', 'stars', 'blackhole', 'GaS', 'blackholes']
    nums = [0, 1, 3, 4, 5, 0, 5]

    for name, num in zip(names, nums):
        pn = ill.snapshot.partTypeNum(name)
        print("\npartTypeNum('{}') = '{}' (should be '{}')".format(name, pn, num))
        assert_equal(pn, num)

    return


def test_snapshot_partTypeNum_2():
    # These should fail and raise an exception
    names = ['peanuts', 'monkeys']
    nums = [0, 1]

    for name, num in zip(names, nums):
        print("\npartTypeNum('{}') should raise `Exception`".format(name))
        assert_raises(Exception, ill.snapshot.partTypeNum, name)

    return
