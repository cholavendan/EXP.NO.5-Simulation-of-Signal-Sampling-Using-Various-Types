# EXP.NO.5-Simulation-of-Signal-Sampling-Using-Various-Types
5.Simulation of Signal Sampling Using Various Types such as
    i) Ideal Sampling
    ii) Natural Sampling
    iii) Flat Top Sampling

# AIM
To simulate and analyze signal sampling techniques including Ideal Sampling, Natural Sampling, and Flat Top Sampling using Python.

# SOFTWARE REQUIRED
Python 3.x

NumPy Library

Matplotlib Library

# ALGORITHMS
1. Ideal Sampling:

Generate a continuous-time message signal.

Create an impulse train with high frequency.

Multiply the message signal with the impulse train.

2. Natural Sampling:

Generate a continuous-time message signal.

Create a pulse train with finite duration pulses.

Multiply the message signal with the pulse train.

3. Flat Top Sampling:

Generate a continuous-time message signal.

Create a pulse train with flat tops.

Multiply the message signal with the pulse train.

# PROGRAM
1. Impulse Sampling
   
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

2. Natural sampling
   
import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter

fs = 1000  # Sampling frequency (samples per second)

T = 1  # Duration in seconds

t = np.arange(0, T, 1/fs) 

fm = 5  # Frequency of message signal (Hz)

message_signal = np.sin(2 * np.pi * fm * t)

pulse_rate = 50 

pulse_train = np.zeros_like(t)

pulse_width = int(fs / pulse_rate / 2)

for i in range(0, len(t), int(fs / pulse_rate)):

    pulse_train[i:i+pulse_width] = 1 # Indented this line
    
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

3. Flat Top Sampling

import numpy as np

import matplotlib.pyplot as plt

fs = 1000  # Sampling frequency (Hz)

f = 5      # Signal frequency (Hz)

duration = 1  # Duration in seconds

sampling_period = 0.05  # Sampling period for flat-top

t = np.linspace(0, duration, int(fs * duration))

signal = np.sin(2 * np.pi * f * t)

samples = []

sampled_time = []

reconstructed_signal = np.zeros_like(signal)

i = 0

while i < len(t):

    end = i + int(sampling_period * fs)
    
    if end > len(t):
    
        end = len(t)
        
    flat_value = np.mean(signal[i:end])  # Flat-top value
    
    samples.extend([flat_value] * (end - i))
    
    sampled_time.extend(t[i:end])
    
    reconstructed_signal[i:end] = flat_value
    
    i = end
    
plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal', linewidth=1)

plt.title('Continuous Signal')

plt.xlabel('Time (s)')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

plt.figure(figsize=(10, 4))

plt.stem(sampled_time[::int(fs * sampling_period)], samples[::int(fs * sampling_period)], linefmt='r--', markerfmt='ro', basefmt='k-', label='Sampled Signal')

plt.title('Flat-top Sampled Signal')

plt.xlabel('Time (s)')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

plt.figure(figsize=(10, 4))

plt.plot(t, reconstructed_signal, 'g-', linewidth=1.5, label='Reconstructed Signal')

plt.title('Reconstructed Signal')

plt.xlabel('Time (s)')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()


   

# OUTPUT
1. Ideal Sampling
![Screenshot 2025-03-23 145459](https://github.com/user-attachments/assets/de16494f-b988-4d24-89b0-72e30e76a9a8)
![Screenshot 2025-03-23 145510](https://github.com/user-attachments/assets/2d1d25cd-583f-4665-8848-5dee257e9959)
2. Natural Sampling
![Screenshot 2025-03-23 145050](https://github.com/user-attachments/assets/1315e4ff-0a96-4f84-9cb7-90045096da53)
![Screenshot 2025-03-23 145116](https://github.com/user-attachments/assets/be01e9fe-b86b-4ba4-b5b8-d033e818d1df)
3. Flat Top Sampling
![Screenshot 2025-03-23 151007](https://github.com/user-attachments/assets/9b6b96e0-3d68-4b4b-9e07-6a34855c4661)
![Screenshot 2025-03-23 151016](https://github.com/user-attachments/assets/55d27f2a-55b5-4603-ab9a-eb2444a5f4a3)




 
# RESULT / CONCLUSIONS
Ideal sampling preserves signal shape but is not practical for real-time systems.

Natural sampling retains signal shape within pulse duration but introduces distortion.

Flat top sampling provides a constant amplitude but suffers from amplitude droop.

Among all, natural sampling closely represents practical analog-to-digital conversion.

