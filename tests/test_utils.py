"""Test for utils module"""

import numpy as np
import pytest

from empytools.utils import time_array
from empytools.utils import freq_array
from empytools.utils import get_si_str
from empytools.utils import slice_arr
from empytools.utils import noise_floor_to_sigma


data_time_array = zip(
    np.random.randint(low=2, high=50, size=10),
    10 ** np.random.normal(loc=1, scale=1, size=10),
)


@pytest.mark.parametrize("n,fs", data_time_array)
def test_time_array(n, fs):
    """Test time_array"""
    result = time_array(fs, n)
    expect = np.arange(n) / fs
    assert np.allclose(result, expect)


data_freq_array = zip(
    np.random.randint(low=2, high=50, size=10),
    10 ** np.random.normal(loc=1, scale=1, size=10),
)


@pytest.mark.parametrize("n,fs", data_freq_array)
def test_freq_array_float(n, fs):
    """Test freq_array"""
    result = freq_array(fs, n)
    expect = (n % 2) * fs / (2 * n) + np.linspace(-fs / 2, fs / 2, n, endpoint=False)
    assert np.allclose(result, expect)


data_get_si_str = [
    (3.265e-3, "3.27 m"),
    (3.264e6, "3.26 M"),
    (3.265e-19, "0.326 a"),
    (3.264e15, "3.26e+03 T"),
    (2e3, "2.00 k"),
    (600e-3, "600. m"),
]


@pytest.mark.parametrize("x,expect", data_get_si_str)
def test_get_si_str(x, expect):
    """Test in/out bounds inputs"""
    result = get_si_str(x)
    assert result == expect


data_test_slice = [
    (np.ones(12), 4, (4, 3)),
    (np.ones((4,3)), 3, (3, 4)),
]


@pytest.mark.parametrize("x,length,expect", data_test_slice)
def test_slice(x, length, expect):
    """Check reshaped array"""
    result = slice_arr(x, length).shape
    assert result == expect


def test_slice_error():
    """Check reshaped array with bad value"""
    x = np.arange(12)
    length = 13
    with pytest.raises(ValueError):
        slice_arr(x, length)


def test_noise_to_sigma():
    """Check noise floor function"""
    result = noise_floor_to_sigma(nf=10, alpha=10)
    expect = 10
    assert result == expect
