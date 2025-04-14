"""TimeFreqData Class"""

import numpy as np
from matplotlib import pyplot as plt
from empytools import utils


class TimeFreqData:
    """
    Class to hold various time and frequency parameters of a given signal

    Independent Attributes
    ----------------------
    x_t  : array-like
           Time domain signal, ifft(X)
    fs   : scalar
           Sampling frequency
    n    : integer
           Number of samples

    Dependent Attributes
    --------------------
    x_f   : array-like
            Frequency domain signal, fft(x)
    p_x   : array-like
            Power in frequency domain, |X|^2
    psd_x : array-like
            Power spectral density estimate
    fbin  : scalar
            Size of a single frequency bin
    t     : array-like
            Time array (for plotting)
    f     : array-like
            Frequency array (for plotting)

    Attribute Relationships
    -----------------------
          x_t --> x_f, p_x
          x --> n (if n not provided)
      fs, n --> fbin, t, f
    x, fbin --> psd_x

    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    # pylint: disable=too-many-branches
    # I don't want to think about how to fix these right now

    def __init__(self, x_t, fs=1, n=None):
        """
        Initialize method.

        Only provide x or X for instantiation, not both.

        Parameters
        ----------
        x_t : array-like
              Input time domain data
        fs  : scalar, optional
              Input sampling frequency
        n   : scalar, optional
              Number of samples for an FFT. If not provided n=length(x), if
              provided then spectral averaging will be used.
        """

        self._t = []
        self._f = []
        self._fbin = []
        self._psd_x = []

        if n is None:
            n = len(x_t)

        self.n = n
        self.fs = fs
        self.x_t = x_t

    # Properties with setters
    @property
    def x_t(self):
        """Return x(t)"""
        return self._x_t

    @x_t.setter
    def x_t(self, x_t):
        """Set x(t) and update dependent paramaters"""
        self._x_t_orig = x_t  # Save the original data in case slicing deletes some
        self._x_t = utils.slice_arr(x_t, self.n)
        self._x_f = np.fft.fftshift(np.fft.fft(self.x_t, axis=0) / self.n)
        self._p_x = np.abs(self.x_f**2)
        self.__update_time_freq(self.fs, self.n)

    @property
    def n(self):
        """Return number of samples"""
        return self._n

    @n.setter
    def n(self, n):
        """Set number of samples and update x(t) if necessary"""
        self._n = n
        if hasattr(self, "_x_t"):
            # x exists and needs to be reshaped
            # Grab original data before reshaping it
            self.x_t = self._x_t_orig

    @property
    def fs(self):
        """Return sample frequency"""
        return self._fs

    @fs.setter
    def fs(self, fs):
        """Set sample frequency and update time/freq arrays"""
        self._fs = fs
        if hasattr(self, "_n") and hasattr(self, "_p_x"):
            self.__update_time_freq(self.fs, self.n)

    # Properties without setters (read only)
    @property
    def x_f(self):
        """Return X(f)"""
        return self._x_f

    @property
    def p_x(self):
        """Return P(x)"""
        return self._p_x

    @property
    def t(self):
        """Retrun time array"""
        return self._t

    @property
    def f(self):
        """Return frequency array"""
        return self._f

    @property
    def fbin(self):
        """Return frequency bin size"""
        return self._fbin

    @property
    def psd_x(self):
        """Return PSD(x)"""
        return self._psd_x

    # Private methods
    def __update_time_freq(self, fs, n):
        """Update parameters depending on fs, n"""
        self._t = utils.time_array(fs, n)
        self._f = utils.freq_array(fs, n)
        self._fbin = fs / n
        self._psd_x = self.p_x / self.fbin

    # Public methods
    def plot_time(self, fmt="-o", avg=True, hold=False, unit="V"):
        """
        Plot time data.

        Parameters
        ----------
        fmt  : string, optional
               PyPlot line format string
        avg  : bool, optional
               Average data across slices before plotting
        hold : bool, optional
               Hold plot for other plotting
        unit : string, optional
               Unit for y-axis label
        """
        t_scale, t_unit = utils.get_si(self.t)
        if avg:
            plt_data = np.mean(self.x_t, axis=1)
        else:
            plt_data = self.x_t

        plt.plot(self.t / t_scale, plt_data, fmt)
        plt.xlabel(f"Time [{t_unit}s]")
        plt.ylabel(f"Amplitude [{unit}]")
        plt.grid(True)
        if not hold:
            plt.show()

    def plot_freq(
        self, fmt="-o", avg=True, hold=False, unit="V", power=False, psd=False, db=False
    ):
        """
        Plot frequency data.

        Parameters
        ----------
        fmt   : string, optional
                PyPlot line format string
        avg   : bool, optional
                Average data across slices before plotting
        hold  : bool, optional
                Hold plot for other plotting
        unit  : string, optional
                Unit for y-axis label
        power : bool, optional
                Plot y-axis in power units
        psd   : bool, optional
                Plot y-axis in power spectral density units
        db    : bool, optional
                Plot y-axis in dB. Requires power=True, or psd=True
        """
        f_scale, f_unit = utils.get_si(self.f)

        if psd:
            if avg:
                plt_data = np.mean(self.psd_x, axis=1)
            else:
                plt_data = self.psd_x

            if db:
                unit = "dBW/Hz"
                plt.plot(self.f / f_scale, 10 * np.log10(plt_data), fmt)
            else:
                unit = "W/Hz"
                plt.plot(self.f / f_scale, plt_data, fmt)
        elif power:
            if avg:
                plt_data = np.mean(self.p_x, axis=1)
            else:
                plt_data = self.p_x

            if db:
                unit = "dBW"
                plt.plot(self.f / f_scale, 10 * np.log10(plt_data), fmt)
            else:
                unit = "W"
                plt.plot(self.f / f_scale, plt_data, fmt)
        else:
            if avg:
                plt_data = np.mean(self.x_f, axis=1)
            else:
                plt_data = self.x_f

            plt.plot(self.f / f_scale, np.abs(plt_data), fmt)

        plt.xlabel(f"Frequency [{f_unit}Hz]")
        plt.ylabel(f"Amplitude [{unit}]")
        plt.grid(True)
        if not hold:
            plt.show()
