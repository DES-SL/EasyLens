__author__ = 'sibirrer'


from easylens.Data.lens_system import LensSystem

import numpy as np
import pytest


class TestLensSystem(object):

    def setup(self):
        self.lensSystem = LensSystem('test', 0, 0)

    def test_get_angle_coord(self):
        assert 0 == 0

    def test_numpy_sqrt(self):
        assert np.sqrt(4) == 2

if __name__ == '__main__':
    pytest.main()