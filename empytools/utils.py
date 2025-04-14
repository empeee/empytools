"""Utilities module"""

import numpy as np


def time_array(fs, n):
    """
    Get time array.

    Calculate an array of time values for a given sample rate and number of
    points.

    Parameters
    ----------
    fs : scalar
         Sampling frequency
    n  : scalar
         Number of samples

    Returns
    -------
    out : array
          Time array
    """
    return np.linspace(0, n / fs, n, endpoint=False)


def freq_array(fs, n):
    """
    Get frequency array.

    Calculate an array of frequency values for a given sample rate and number
    of points.

    Parameters
    ----------
    fs : scalar
         Sampling frequency
    n  : scalar
         Number of samples

    Returns
    -------
    out : array
          Frequency array
    """
    return np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))


def get_si(x_arr):
    """
    Get unit prefix and label for an array.

    Returns a scaling factor and SI-prefix string for a given input array.

    Parameters
    ----------
    x_arr : array-like
            Input array

    Returns
    -------
    x_scale : scalar
              Scalar for x_arr corresponding to the SI-prefix for the max
              element
    x_unit  : string
              String of SI-prefix corresponding to the max element in x_arr
    """
    units = {
        -18: "a",
        -15: "f",
        -12: "p",
        -9: "n",
        -6: "u",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
    }
    x_max = np.max(x_arr)
    x_order = np.log10(x_max)
    x_sign = np.sign(x_order)
    x_scale = x_sign * np.floor(np.abs(x_order) / 3) * 3
    x_scale = np.floor(x_order / 3) * 3
    # Bound to defined orders
    x_scale = np.max((np.min((x_scale, 12)), -18))
    return 10**x_scale, units[x_scale]


def get_si_str(x):
    """
    Return string of value scaled to SI prefix for pretty printing values

    Parameters
    ----------
    x : scalar
        Input value

    Returns
    -------
    x_si_str : str
               String that is x/scale with SI prefix
    """
    scale, unit = get_si(x)
    return f"{x/scale:#.3g} {unit}"


def slice_arr(x, length):
    """
    Slices 1-D array and reshapes into 2-D array.

    Slices 1-D array, 'x', into 'length' sized chunks reshapes into 2-D array.
    Unused data is discarded.

    Parameters
    ----------
    x      : array-like
             1-D input array
    length : scalar
             Size of slice

    Returns
    -------
    x_slice : array-like
              2-D output array
    """
    if len(x.shape) != 1:
        # x has already been reshaped and we need to reset it. Hopefully it was
        # reshaped with this function or things might get wonky
        x = x.T.reshape(-1)

    if length > x.size:
        raise ValueError(
            f"length ({length}) cannot be greater than the size of x ({x.size})"
        )

    num = int(np.floor(x.size / length))
    x_slice = x[0 : num * length]
    x_slice = x_slice.reshape((num, length)).T

    return x_slice


def noise_floor_to_sigma(nf, alpha):
    """
    Calculates the sigma of a normally distributed random variable that would
    result in a specific noise floor.

    Can be used to calculate noise floor for both power spectrum and power
    spectral density (PSD). The value of the parameter alpha depends on which
    mode is being used.

    Parameters
    ----------
    nf    : scalar
            Desired noise floor in dB (dBW or dBW/Hz)
    alpha : scalar
            if psd=False: alpha = Number of samples
            if psd=True:  alpha = Sample frequency

    Returns
    -------
    sigma : scalar
            Sigma value of normally distributed random variable that will
            result in the desired noise floor
    """
    return np.sqrt(alpha * 10 ** (nf / 10))
