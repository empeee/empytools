import numpy as np
import pytest

from empytools.utils import time_array
from empytools.utils import freq_array
from empytools.utils import get_si_str
from empytools.utils import slice_arr
from empytools.utils import noise_floor_to_sigma

def test_time_array():
    """Test time_array"""
    for i in range(0,10):
        N = np.random.randint(low=2, high=100)
        fs = 10**np.random.normal(loc=1, scale=10)
        result = time_array(fs, N)
        expect = np.arange(N) / fs
        assert np.allclose(result, expect)

def test_freq_array_float():
    """Test freq_array"""
    for i in range(0,10):
        N = np.random.randint(low=2, high=50)
        fs = 10**np.random.normal(loc=1, scale=1)
        result = freq_array(fs, N)
        expect = (N%2)*fs/(2*N) + np.linspace(-fs / 2, fs / 2, N, endpoint=False)
        assert np.allclose(result, expect)

def test_get_si_str():
    """Test in/out bounds inputs"""
    x = 3.265e-3
    result = get_si_str(x)
    expect = "3.27 m"
    assert result == expect

    x = 3.264e6
    result = get_si_str(x)
    expect = "3.26 M"
    assert result == expect

    x = 3.265e-19
    result = get_si_str(x)
    expect = "0.326 a"
    assert result == expect

    x = 3.264e15
    result = get_si_str(x)
    expect = "3.26e+03 T"
    assert result == expect

def test_slice():
    """Check reshaped array"""
    x = np.arange(12)
    length = 4
    x_slice = slice_arr(x, length) 
    result = x_slice.shape
    expect = (4, 3)
    assert result == expect

    # Reshape
    x = x_slice
    length = 3
    result = slice_arr(x, length).shape
    expect = (3, 4)
    assert result == expect

def test_slice_error():
    """Check reshaped array with bad value"""
    x = np.arange(12)
    length = 13
    with pytest.raises(ValueError):
        result = slice_arr(x, length)


def test_noise_to_sigma():
    """Check noise floor function"""
    result = noise_floor_to_sigma(nf=10, alpha=10)
    expect = 10
    assert result == expect
