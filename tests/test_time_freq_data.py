import numpy as np
from matplotlib import pyplot as plt
import pytest

from empytools.time_freq_data import TimeFreqData

def test_class_init():
    x = np.array([0, 1, 0, -1])
    fs = 1

    x_expect = np.array(x).reshape((4, 1))
    fs_expect = fs
    n_expect = len(x)
    x_f_expect = np.array([0, 0.5, 0, 0.5]).reshape((4,1))
    p_x_expect = x_f_expect**2
    fbin_expect = fs_expect / n_expect
    psd_x_expect = p_x_expect / fbin_expect
    t_expect = np.arange(n_expect) / fs_expect
    f_expect = (n_expect % 2) * fs_expect / (2 * n_expect) + \
                np.linspace(-fs_expect / 2, fs_expect / 2, n_expect, 
                            endpoint=False)

    d1 = TimeFreqData(x, fs)
    
    assert np.allclose(d1.x_t, x_expect)
    assert d1.fs == fs_expect
    assert d1.n == n_expect
    assert np.allclose(np.abs(d1.x_f), x_f_expect)
    assert np.allclose(d1.p_x, p_x_expect)
    assert d1.fbin == fbin_expect
    assert np.allclose(d1.psd_x, psd_x_expect)
    assert np.allclose(d1.t, t_expect)
    assert np.allclose(d1.f, f_expect)

def test_n_setter():
    x = np.array([0, 1, 0, -1, 0, 1, 0, -1])

    d1 = TimeFreqData(x)
    d1.n = 4
    assert d1.x_t.shape == (4, 2)

def test_fs_setter():
    x = np.array([0, 1, 0, -1])
    fs = 1e3
    n = len(x)
    
    d1 = TimeFreqData(x)
    d1.fs = fs

    assert d1.fbin == fs/n

def test_plots():
    x = np.array([1, 1, 0, -1])
    d1 = TimeFreqData(x)

    d1.plot_time()
    d1.plot_time(avg=False)
    d1.plot_freq()
    d1.plot_freq(avg=False)
    d1.plot_freq(psd=True)
    d1.plot_freq(psd=True, avg=False)
    d1.plot_freq(psd=True, db=True)
    d1.plot_freq(power=True)
    d1.plot_freq(power=True, avg=False)
    d1.plot_freq(power=True, db=True)