import time
import numpy as np
import matplotlib.pyplot as plt
import pyuda
client=pyuda.Client()
from scipy.signal import find_peaks

class ELM_signal(object):

    def __init__(self,raw_signal,raw_time,window_size_time=5e-3):
        """
        raw_signal: 1D array e.g. dalpha
        raw_time: 1D array e.g. dalpha time axis
        window_size_time: smoothing window size in units of seconds
        """
        running_mean=[]
        running_std=[]
        window_size_steps=int(np.ceil(window_size_time/np.mean(np.diff(raw_time))))
        number_steps=int(len(raw_signal)-window_size_steps)
        time_window=[]
        for ii in range(number_steps):
            time_window.append(np.nanmean(raw_time[ii:ii+window_size_steps]))
            running_mean.append(np.nanmean(raw_signal[ii:ii+window_size_steps]))
            running_std.append(np.nanstd(raw_signal[ii:ii+window_size_steps]))
        time_window=np.array(time_window)
        running_mean=np.array(running_mean)
        running_std=np.array(running_std)
        ELM_signal.raw_signal=raw_signal
        ELM_signal.raw_time=raw_time
        ELM_signal.smooth_mean=running_mean
        ELM_signal.smooth_std=running_std
        ELM_signal.smooth_time=time_window

    def plot_smoothed_signal(self,show=True):
        plt.figure()
        plt.plot(ELM_signal.raw_time,ELM_signal.raw_signal,label='raw')
        plt.plot(ELM_signal.smooth_time,ELM_signal.smooth_mean,label='smooth')
        plt.xlim(0.5,0.6)
        plt.xlabel('Time (s)')
        plt.ylabel('ELM signal')
        plt.legend()
        plt.grid()
        if show==True:
            plt.show()
            plt.savefig("plots/plot_smoothed_signal.png")
    def normalise_signal(self):
        start_index=np.argmin(np.abs(ELM_signal.raw_time-ELM_signal.smooth_time[0]))
        ELM_signal_cut=ELM_signal.raw_signal[start_index:start_index+len(ELM_signal.smooth_time)]
        ELM_signal_norm_time=ELM_signal.raw_time[start_index:start_index+len(ELM_signal.smooth_time)]
        ELM_signal_norm=(ELM_signal_cut-ELM_signal.smooth_mean)/ELM_signal.smooth_std
        ELM_signal.norm_signal=ELM_signal_norm
        ELM_signal.ac_signal=ELM_signal.raw_signal[start_index:start_index+len(ELM_signal.smooth_time)]-ELM_signal.smooth_mean
    def find_ELM_times_norm(self,norm_thres=2.5,min_time_peaks=0.5e-3):
        h_distance_steps=int(min_time_peaks/np.mean(np.diff(ELM_signal.smooth_time)))
        peaks, _ = find_peaks(ELM_signal.norm_signal, height=norm_thres,distance=h_distance_steps)
        ELM_signal.ELM_norm_times=ELM_signal.smooth_time[peaks]
        ELM_signal.ELM_norm_values=ELM_signal.norm_signal[peaks]
    def find_ELM_times_ac(self,ac_thres=0.02,min_time_peaks=0.5e-3):
        h_distance_steps=int(min_time_peaks/np.mean(np.diff(ELM_signal.smooth_time)))
        peaks, _ = find_peaks(ELM_signal.ac_signal, height=ac_thres,distance=h_distance_steps)
        ELM_signal.ELM_ac_times=ELM_signal.smooth_time[peaks]
        ELM_signal.ELM_ac_values=ELM_signal.ac_signal[peaks]
    def plot_normalised_signal(self,scale_raw=30,show=True):
        plt.figure()
        plt.plot(ELM_signal.smooth_time,ELM_signal.norm_signal,label='Normalised signal')
        plt.plot(ELM_signal.raw_time,scale_raw*ELM_signal.raw_signal,label='x'+str(scale_raw)+' Raw signal')
        plt.xlim(0.5,0.6)
        plt.xlabel('Time (s)')
        plt.ylabel('ELM signal')
        plt.legend()
        plt.grid()
        try:
            plt.scatter(ELM_signal.ELM_norm_times,ELM_signal.ELM_norm_values,color='r')
        except:
            print("Have not found ELM times yet. Run ELM_signal.find_ELM_times_norm()")
        if show==True:
            plt.show()
            plt.savefig("plots/plot_normalised_signal.png")
    def plot_ac_signal(self,scale_raw=0.5,show=True):
        plt.figure()
        plt.plot(ELM_signal.smooth_time,ELM_signal.ac_signal,label='Mean subtracted')
        plt.plot(ELM_signal.raw_time,scale_raw*ELM_signal.raw_signal,label='x'+str(scale_raw)+' Raw signal')
        plt.xlim(0.5,0.6)
        plt.xlabel('Time (s)')
        plt.ylabel('ELM signal')
        plt.legend()
        plt.grid()
        try:
            plt.scatter(ELM_signal.ELM_ac_times,ELM_signal.ELM_ac_values,color='r')
        except:
            print("Have not found ELM times yet. Run ELM_signal.find_ELM_times_ac()")
        if show==True:
            plt.show()
            plt.savefig("plots/plot_ac_signal.png")
if __name__ == '__main__':
    shot=46631
    start_time = time.time()
    print("running")
    dalpha=client.get('/xim/da/hm10/t',shot)

    print("got client data")
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.plot(dalpha.time.data, dalpha.data)
    plt.xlim(0.5,0.6)
    plt.xlabel("time (s)")
    plt.ylabel("dalpha trace")
    plt.savefig("plots/dalpha.png")
    ELM_signal=ELM_signal(dalpha.data,dalpha.time.data)
    ELM_signal.plot_smoothed_signal()
    ELM_signal.normalise_signal()
    print("--- %s seconds ---" % (time.time() - start_time))

    #Method 1: subtract running mean from signal and search for peaks
    ELM_signal.find_ELM_times_ac(ac_thres=0.02,min_time_peaks=0.5e-3)
    ELM_signal.plot_ac_signal()
    print(ELM_signal.ELM_ac_times) #time of ELMs
    print(len(ELM_signal.ELM_ac_times))
    print("--- %s seconds ---" % (time.time() - start_time))

    #Method 2: normalise signal and then search for peaks (signal-mean)/std
    ELM_signal.find_ELM_times_norm(norm_thres=2.3,min_time_peaks=0.5e-3)
    ELM_signal.plot_normalised_signal()
    print(ELM_signal.ELM_norm_times) #time of ELMs
    print(len(ELM_signal. ELM_norm_times))
    print("--- %s seconds ---" % (time.time() - start_time))
    print("done")
