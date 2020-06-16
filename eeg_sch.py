import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from scipy.integrate import simps
import matplotlib.pyplot as plt
import glob, os

n_ch=16
sampling_hz=128.0
samples=int(data.size/n_ch)
time=np.arange(data.size/n_ch)/sampling_hz


def frame(way):
    os.chdir(way)
    for file in glob.glob("*.txt"):
        data=np.loadtxt(file)
        data.resize(n_ch,samples)
        data_frame=pd.DataFrame(np.transpose(data), columns=['F7','F3','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','P6','O1','O2'],
                               index=None)
        return data_frame
   
    """f0=50.0 #should be removed from eeg data #filter
    Q=30.0 #quality factor
    b,a=signal.iirnotch(f0/fs,Q)

    freq,h=signal.freqz(b,a) #frequency response

    fig,ax=plt.subplots(1,1, figsize=(8,6))
    ax[0].plot(freq, 20*np.log10(abs(h)),color='blue')
    ax[0].set_title("Frequency response")
    ax[0].set_ylabel("Amplitude (bB)", color='blue')
    ax[0].set_xlim([0,100])
    ax[0].set_ylim([-25,10])
    ax[0].grid()

    plt.show()""" 

def visual(data_frame):

    #Plot EEG raw signal in mkV
    sns.set(font_scale=1.2)
    
    fig,ax=plt.subplots(1,1, figsize=(12,6))
    plt.plot(time, data_frame, lw=1.5, color='k')
    plt.title('Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('EEG signal (mkV)')
    plt.xlim(time.min(), time.max())
    plt.grid()
    plt.show()
    sns.despine()
    return
    
def power(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

        # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

        # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

        # Frequency resolution
    freq_res = freqs[1] - freqs[0]

        # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
 

    # Plot the power spectrum
    sns.set(font_scale=1.2, style='white')
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, psd, color='k', lw=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (V^2 / Hz)')
    plt.ylim([0, psd.max() * 1.1])
    plt.title("Welch's periodogram")
    plt.xlim([0, freqs.max()])
    sns.despine()

    # Find intersecting values in frequency vector
    idx_delta = np.logical_and(freqs >= low, freqs <= high)

    # Plot the power spectral density and fill the delta area
    plt.figure(figsize=(7, 4))
    plt.plot(freqs, psd, lw=2, color='k')
    plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (uV^2 / Hz)')
    plt.xlim([0, 10])
    plt.ylim([0, psd.max() * 1.1])
    plt.title("Delta")
    sns.despine()

    return print("Power of EEG signal is", bp)
  


