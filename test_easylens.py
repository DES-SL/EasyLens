
# Copyright (C) 2015 ETH Zurich, Institute for Astronomy
# Distributed under MIT license

"""
Tests for `EasyLens` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import pytest
import easylens


class TestEasyLens(object):

    def setup(self):
        #prepare unit test. Load data etc
        print("setting up " + __name__)

    def test_something(self):
        x = 1
        assert x==1

    def teardown(self):
        #tidy up
        print("tearing down " + __name__)

if __name__ == '__main__':
    pytest.main()
