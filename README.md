# EXP.NO.5-Simulation-of-Signal-Sampling-Using-Various-Types
5.Simulation of Signal Sampling Using Various Types such as
    i) Ideal Sampling
    ii) Natural Sampling
    iii) Flat Top Sampling

# AIM
To perform and verify Ideal sampling,Natural sampling and Flat Top sampling using python.

# SOFTWARE REQUIRED
Python

# ALGORITHMS

1. Import Libraries and Define Original Signal:
Import necessary libraries: numpy and matplotlib.pyplot. Define original signal parameters: sampling frequency, time array, signal frequency, and signal amplitude.

2. Define Sampling Parameters:
Define sampling frequency and time array for sampling the original signal.

3. Sample Original Signal:
Sample the original signal using the defined sampling parameters to obtain the sampled signal.

4. Reconstruct Sampled Signal:
Reconstruct the sampled signal using a reconstruction technique, such as zero-order hold or linear interpolation.

5. Plot Results:
Plot the original signal, sampled signal, and reconstructed signal using matplotlib.pyplot to visualize the results.
  


# PROGRAM

Ideal Sampling:

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import resample

fs = 100

t = np.arange(0, 1, 1/fs)

f = 5

signal = np.sin(2 * np.pi * f * t)

plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal')

plt.title('Continuous Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

t_sampled = np.arange(0, 1, 1/fs)

signal_sampled = np.sin(2 * np.pi * f * t_sampled)

plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal', alpha=0.7)

plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal (fs = 100 Hz)')

plt.title('Sampling of Continuous Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

reconstructed_signal = resample(signal_sampled, len(t))

plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal', alpha=0.7)

plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')

plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()


Natural Sampling:

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter

fs = 1000 

T = 1  

t = np.arange(0, T, 1/fs)

fm = 5  

message_signal = np.sin(2 * np.pi * fm * t)

pulse_rate = 50 

pulse_train = np.zeros_like(t)

pulse_width = int(fs / pulse_rate / 2)

for i in range(0, len(t), int(fs / pulse_rate)):

pulse_train[i:i+pulse_width] = 1

nat_signal = message_signal * pulse_train

sampled_signal = nat_signal[pulse_train == 1]

sample_times = t[pulse_train == 1]

reconstructed_signal = np.zeros_like(t)

for i, time in enumerate(sample_times):

    index = np.argmin(np.abs(t - time))
    
    reconstructed_signal[index:index+pulse_width] = sampled_signal[i]
    
def lowpass_filter(signal, cutoff, fs, order=5):

nyquist = 0.5 * fs

normal_cutoff = cutoff / nyquist

b, a = butter(order, normal_cutoff, btype='low', analog=False)

return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal,10, fs)

plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)

plt.plot(t, message_signal, label='Original Message Signal')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 2)

plt.plot(t, pulse_train, label='Pulse Train')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 3)

plt.plot(t, nat_signal, label='Natural Sampling')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 4)

plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


Flat Top Sampling:

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter

fs = 1000 

T = 1  

t = np.arange(0, T, 1/fs) 


fm = 5 

message_signal = np.sin(2 * np.pi * fm * t)

pulse_rate = 50  

pulse_train = np.zeros_like(t)

pulse_width = int(fs / pulse_rate / 4)

for i in range(0, len(t), int(fs / pulse_rate)):

    pulse_train[i:i+pulse_width] = 1

flat_top_signal = np.copy(message_signal)

for i in range(0, len(t), int(fs / pulse_rate)):

    flat_top_signal[i:i+pulse_width] = message_signal[i]  
    
sampled_signal = flat_top_signal[pulse_train == 1]

sample_times = t[pulse_train == 1]

reconstructed_signal = np.zeros_like(t)

for i, time in enumerate(sample_times):

    index = np.argmin(np.abs(t - time))
    
    reconstructed_signal[index:index+pulse_width] = sampled_signal[i]
    
def lowpass_filter(signal, cutoff, fs, order=5):

    nyquist = 0.5 * fs
    
    normal_cutoff = cutoff / nyquist
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal, 10, fs)

plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)

plt.plot(t, message_signal, label='Original Message Signal')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 2)

plt.plot(t, pulse_train, label='Pulse Train')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 3)

plt.plot(t, flat_top_signal, label='Flat-Top Sampled Signal')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 4)

plt.plot(t, reconstructed_signal, label='Reconstructed Signal', color='green')

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# OUTPUT

Ideal Sampling:
![Screenshot 2025-03-23 145459](https://github.com/user-attachments/assets/95a3c70c-36b4-4c53-89f6-721c232c31c4)
![Screenshot 2025-03-23 145510](https://github.com/user-attachments/assets/31295122-605f-4e62-8e88-1608f891d484)

Natural Sampling:
![natural sampling](https://github.com/user-attachments/assets/f0b92e33-603c-4274-8fcd-26fe725dcda1)

Flat Top Sampling:
![Flat top](https://github.com/user-attachments/assets/054abf9d-37d0-46d4-95c2-957fc5674d39)

 
# RESULT / CONCLUSIONS
Thus the Ideal sampling,Natural sampling and Flat Top sampling are succesfully verified using python.
