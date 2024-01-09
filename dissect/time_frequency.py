"""Extraction of the time-frequency features using continuous wavelet transform.

N.B. In the article I used scripts leveraging the R implementation of
the continuous wavelet transform to extract the time frequency features.
Here we use ssqueezepy implementation to obtain a pure python pipeline. As a
result, the extact values of the resulting coefficient might be slightly
different from the ones used in the paper.

Author: Romain FAYAT, January 2024
"""
from ssqueezepy import cwt
from dissect.data_loading import COL_GYR, COL_ACC_R, SR
from dissect.utils import split_and_map
from ssqueezepy.ssqueezing import _ssq_freqrange as get_freqrange
import numpy as np
import pandas as pd


def compute_cwt(x, wavelet="morlet", sr=300., nv=20):
    """Compute the continuous wavelet transform of a 1D signal with logarithmically-spaced frequencies.
    
    This is simply a wrapper for functions available in ssqueezepy.
    
    Inputs
    ------
    x: array, shape=(n_samples,)
        The input signal.
        
    wavelet: str (default: 'morlet')
        A valid wavelet, see ssqueezepy.Wavelet.
    
    sr: float (default: 300.)
        The sampling rate of the signal.
    
    nv: int (default: 20)
        Number of voices (wavelets per octave).
        
    Returns
    -------
    W: array, shape=(n_samples, n_wavelet)
        The CWT transform of the signal.
        
    f: array, shape=(n_wavelet,)
        The frequencies associated with columns of W.

    """
    # Wavelet transform
    W, scales = cwt(x, wavelet=wavelet, fs=sr,  nv=nv, scales="log")
    fm, fM = get_freqrange("peak", 1 / sr, len(x), wavelet, scales, True)
    
    na = len(scales)
    f = fm * np.power(fM / fm, np.arange(na)/(na - 1))
    
    return W[::-1, :].T, f


def get_linearly_spaced_cwt(x, fmin=2.5, fmax=20, nbins=7, **kwargs):
    """Return the average magnitude of CWT transform of x in linearly spaced bands.
    
    Inputs
    ------
    x: Series or 1-dimensional array, shape=(n_samples,)
        The input signal
        
    fmin, fmax: floats (default: 2.5, 20.)
        The minimal and maximal frequency boundaries in Hertz.
    
    nbins: int (default: 7)
        Number of frequency bins
    
    **kwargs, key-word arguments passed to compute_cwt
    
    Returns
    -------
    out, DataFrame, shape=(n_samples, nbins)
        The averaged magnitude of CWT coefficients for x.
    
    """
    if isinstance(x, pd.Series):
        index = x.index.copy()
        prepend = x.name + "_"
        x = x.values
    elif isinstance(x, np.ndarray) and x.ndim == 1:
        index = None
        prepend = ""
    else:
        raise ValueError("Input must be pd.Series or 1d array.")
    
    W, f = compute_cwt(x, **kwargs)
    W_abs = np.abs(W)
    
    bins = np.linspace(fmin, fmax, nbins + 1)    
    out = pd.DataFrame(index=index)
    
    for i, (f_low, f_high) in enumerate(zip(bins[:-1], bins[1:])):
        col = f"{prepend}{f_low:.1f}-{f_high:.1f}"
        is_selected = (f >= f_low) & (f < f_high)
        out[col] = np.mean(W_abs[:, is_selected], axis=1)
    return out


def compute_feature_cwt(imu, chpt, sr=SR, col_acc_R=COL_ACC_R, col_gyr=COL_GYR):
    "Compute the binned wavelet coefficient per frequency (average) and per segment (median)."
    out = pd.DataFrame(index=np.arange(len(chpt)))
    
    for c in col_acc_R + col_gyr:
        cwt_linear = get_linearly_spaced_cwt(imu[c], fmin=2.5, fmax=20, nbins=7, sr=sr)
        cwt_window = split_and_map(cwt_linear, chpt[:-1],
                                   lambda x: np.median(x, axis=0),
                                   n_jobs=None)
        out = out.join(pd.DataFrame(cwt_window, columns=cwt_linear.columns))
        
    return np.log10(out)
