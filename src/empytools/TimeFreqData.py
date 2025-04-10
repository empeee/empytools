import numpy as np
from matplotlib import pyplot as plt
import empytools.utils as utils


class TimeFreqData:
    """
    Class to hold various time and frequency parameters of a given signal

    Independent Attributes
    ----------------------
    x    : array-like
           Time domain signal, ifft(X)
    fs   : scalar
           Sampling frequency
    N    : integer
           Number of samples

    Dependent Attributes
    --------------------
    X    : array-like
           Frequency domain signal, fft(x)
    Px   : array-like
           Power in frequency domain, |X|^2
    PSDx : array-like
           Power spectral density estimate
    fbin : scalar
           Size of a single frequency bin
    t    : array-like
           Time array (for plotting)
    f    : array-like
           Frequency array (for plotting)

    Attribute Relationships
    -----------------------
          x --> X, Px
          x --> N (if N not provided)
      fs, N --> fbin, t, f
    x, fbin --> PSDx

    """

    def __init__(self, x, fs=1, N=None):
        """
        Initialize method.

        Only provide x or X for instantiation, not both.

        Parameters
        ----------
        x  : array-like
             Input time domain data
        fs : scalar, optional
             Input sampling frequency
        N  : scalar, optional
             Number of samples for an FFT. If not provided N=length(x), if
             provided then spectral averaging will be used.
        """
        if N is None:
            N = len(x)

        self.N = N
        self.fs = fs
        self.x = x

    # Properties with setters
    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._xorig = x  # Save the original data in case slicing deletes some
        self._x = utils.slice(x, self.N)
        self._X = np.fft.fftshift(np.fft.fft(self.x, axis=0) / self.N)
        self._Px = np.abs(self.X**2)
        self.__update_time_freq(self.fs, self.N)

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        self._N = N
        if hasattr(self, "_x"):
            # x exists and needs to be reshaped
            # Grab original data before reshaping it
            self.x = self._xorig

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, fs):
        self._fs = fs
        if hasattr(self, "_N") and hasattr(self, "_Px"):
            self.__update_time_freq(self.fs, self.N)

    # Properties without setters (read only)
    @property
    def X(self):
        return self._X

    @property
    def Px(self):
        return self._Px

    @property
    def t(self):
        return self._t

    @property
    def f(self):
        return self._f

    @property
    def fbin(self):
        return self._fbin

    @property
    def PSDx(self):
        return self._PSDx

    # Private methods
    def __update_time_freq(self, fs, N):
        """Update parameters depending on fs, N"""
        self._t = utils.time_array(fs, N)
        self._f = utils.freq_array(fs, N)
        self._fbin = fs/N
        self._PSDx = self.Px / self.fbin

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
            plt_data = np.mean(self.x, axis=1)
        else:
            plt_data = self.x

        plt.plot(self.t / t_scale, plt_data, fmt)
        plt.xlabel(f"Time [{t_unit}s]")
        plt.ylabel(f"Amplitude [{unit}]")
        plt.grid(True)
        if not hold:
            plt.show()

    def plot_freq(self, fmt="-o", avg=True, hold=False, unit="V", power=False,
                  psd=False, db=False):
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
                plt_data = np.mean(self.PSDx, axis=1)
            else:
                plt_data = self.PSDx

            if db:
                unit = "dBW/Hz"
                plt.plot(self.f / f_scale, 10*np.log10(plt_data), fmt)
            else:
                unit = "W/Hz"
                plt.plot(self.f / f_scale, plt_data, fmt)
        elif power:
            if avg:
                plt_data = np.mean(self.Px, axis=1)
            else:
                plt_data = self.Px

            if db:
                unit = "dBW"
                plt.plot(self.f / f_scale, 10*np.log10(plt_data), fmt)
            else:
                unit = "W"
                plt.plot(self.f / f_scale, plt_data, fmt)
        else:
            if avg:
                plt_data = np.mean(self.X, axis=1)
            else:
                plt_data = self.X

            plt.plot(self.f / f_scale, np.abs(plt_data), fmt)

        plt.xlabel(f"Frequency [{f_unit}Hz]")
        plt.ylabel(f"Amplitude [{unit}]")
        plt.grid(True)
        if not hold:
            plt.show()
