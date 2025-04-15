"""Tests for time_freq_data module"""
import pytest
import numpy as np
from matplotlib import pyplot as plt
from empytools.time_freq_data import TimeFreqData


data_tfd = [
    {
        '_t': np.array([0, 1, 2, 3]),
        '_f': np.array([-0.5, -0.25,  0,  0.25]),
        '_fbin': 0.25,
        '_psd_x': np.array([0, 1, 0, 1]).reshape((4,1)),
        '_n': 4,
        '_fs': 1,
        '_x_t_orig': np.array([0, 1, 0, -1]).reshape((4,1)),
        '_x_t': np.array([0, 1, 0,-1]).reshape((4,1)),
        '_x_f': np.array([0+0j, 0+0.5j, 0+0j, 0-0.5j]).reshape((4,1)),
        '_p_x': np.array([0, 0.25, 0, 0.25]).reshape((4,1))
    },
    {
        '_t': np.array([0, 0.001, 0.002, 0.003]),
        '_f': np.array([-500, -250,  0,  250]),
        '_fbin': 250,
        '_psd_x': np.array([0, 0, 0.004, 0]).reshape((4,1)),
        '_n': 4,
        '_fs': 1000,
        '_x_t_orig': np.array([1, 1, 1, 1]).reshape((4,1)),
        '_x_t': np.array([1, 1, 1, 1]).reshape((4,1)),
        '_x_f': np.array([0+0j, 0+0j, 1+0j, 0+0j]).reshape((4,1)),
        '_p_x': np.array([0, 0, 1, 0]).reshape((4,1))
    },
    {
        '_t': np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        '_f': np.array([-0.5, -0.375, -0.25,  -0.125, 0,  0.125, 0.25, 0.375]),
        '_fbin': 0.125,
        '_psd_x': np.array([0, 0, 2, 0, 0, 0, 2, 0]).reshape((8,1)),
        '_n': 8,
        '_fs': 1,
        '_x_t_orig': np.array([0, 1, 0, -1, 0, 1, 0, -1]).reshape((8,1)),
        '_x_t': np.array([0, 1, 0, -1, 0, 1, 0, -1]).reshape((8,1)),
        '_x_f': np.array([0+0j, 0+0j, 0+0.5j, 0+0j, 0+0j, 0+0j, 0-0.5j, 0+0j]).reshape((8,1)),
        '_p_x': np.array([0, 0, 0.25, 0, 0, 0, 0.25, 0]).reshape((8,1))
    },
]


@pytest.mark.parametrize("data", data_tfd)
def test_class_init(data):
    """Check initialization method"""
    d1 = TimeFreqData(data['_x_t'], data['_fs']).__dict__

    for key, val in data.items():
        if type(val) == type(np.array([])):
            assert np.allclose(val, d1[key]), f"{val}, {d1[key]}"
        else:
            assert val == d1[key], f"{val}, {d1[key]}"



@pytest.mark.parametrize("data", data_tfd)
def test_fs_setter(data):
    """Check fs setter method"""
    d1 = TimeFreqData(data['_x_t'])
    d1.fs = data["_fs"]

    assert d1.fbin == data["_fs"] / data["_n"]

data_tfd_reshape = [
    {
        '_t': np.array([0, 1, 2 ,3]),
        '_f': np.array([-0.5,  -0.25, 0, 0.25]),
        '_fbin': 0.25,
        '_psd_x': np.array([[0, 0],[0, 0],[4, 4],[0, 0]]),
        '_n': 4,
        '_fs': 1,
        '_x_t_orig': np.array([1, 1, 1, 1, 1, 1, 1, 1]),
        '_x_t': np.array([[1, 1],[1, 1],[1, 1],[1, 1]]),
        '_x_f': np.array([[0+0j, 0+0j],[0+0j, 0+0j],[1+0j,  1+0j],[0+0j, 0+0j]]),
        '_p_x': np.array([[0, 0],[0, 0],[1, 1],[0, 0]])
    },
    {
        '_t': np.array([0, 1, 2, 3]),
        '_f': np.array([-0.5, -0.25,  0,  0.25]),
        '_fbin': 0.25,
        '_psd_x': np.array([[0, 0], [1, 1],[0, 0],[1, 1]]),
        '_n': 4,
        '_fs': 1,
        '_x_t_orig': np.array([0, 1, 0, -1, 0, 1, 0, -1]),
        '_x_t': np.array([[0, 0],[ 1, 1],[0, 0],[-1, -1]]),
        '_x_f': np.array([[0+0j, 0+0j],[0+0.5j, 0+0.5j],[0+0j , 0+0j ],[0-0.5j, 0-0.5j]]),
        '_p_x': np.array([[0, 0],[0.25, 0.25],[0, 0],[0.25, 0.25]])
    },
]


@pytest.mark.parametrize("data", data_tfd_reshape)
def test_n_setter(data):
    """Check n setter method"""
    d1 = TimeFreqData(data['_x_t_orig'])
    d1.n = d1.n//2
    d1_dict = d1.__dict__
    
    for key, val in data.items():
        if type(val) == type(np.array([])):
            assert np.allclose(val, d1_dict[key]), f"{val}, {d1_dict[key]}"
        else:
            assert val == d1_dict[key], f"{val}, {d1_dict[key]}"

def test_plots(monkeypatch):
    """Check plotting methods"""
    monkeypatch.setattr(plt, 'show', lambda: None)

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
