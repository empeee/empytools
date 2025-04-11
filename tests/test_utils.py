import unittest
import numpy as np

from empytools.utils import time_array
from empytools.utils import freq_array
from empytools.utils import get_si_str
from empytools.utils import slice as mpt_slice


class TestUtils(unittest.TestCase):
    def test_time_array_float(self):
        """
        Test time_array
        """
        fs = 1e3
        N = 10
        result = time_array(fs, N)
        expect = np.arange(N) / fs
        self.assertTrue(np.allclose(result, expect))

    def test_freq_array_float(self):
        """
        Test freq_array
        """
        fs = 1e3
        N = 10
        result = freq_array(fs, N)
        expect = np.linspace(-fs / 2, fs / 2, N, endpoint=False)
        self.assertTrue(np.allclose(result, expect))

    def test_get_si_str_in(self):
        """
        Test in bounds input
        """
        x = 3.265e-3
        result = get_si_str(x)
        expect = "3.27 m"
        self.assertEqual(result, expect)

        x = 3.264e6
        result = get_si_str(x)
        expect = "3.26 M"
        self.assertEqual(result, expect)

    def test_get_si_str_out(self):
        """
        Test out of bounds input
        """
        x = 3.265e-19
        result = get_si_str(x)
        expect = "0.326 a"
        self.assertEqual(result, expect)

        x = 3.264e15
        result = get_si_str(x)
        expect = "3.26e+03 T"
        self.assertEqual(result, expect)

    def test_slice(self):
        """
        Check reshaped array
        """
        x = np.arange(12)
        length = 4
        result = mpt_slice(x, length).shape
        expect = (4, 3)
        self.assertEqual(result, expect)

        # Reshape
        length = 3
        result = mpt_slice(x, length).shape
        expect = (3, 4)
        self.assertEqual(result, expect)


if __name__ == "__main__":
    unittest.main()
