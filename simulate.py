import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from lif_neuron import LIFNeuron

# Simulation parameters
time = 200 # ms
dt = 1.0
steps = int(time / dt)

# Input current (constant)
input_current = 15

# Create neuron
neuron = LIFNeuron(dt=dt)

# Storage
voltages = []
spikes = []

# Run simulation
for t in range(steps):
    v, spike = neuron.step(input_current)
    voltages.append(v)
    spikes.append(spike)

# Time axis
time_axis = np.arange(0, time, dt)

# Plot membrane potential
plt.figure()
plt.plot(time_axis, voltages)
plt.axhline(neuron.v_threshold, linestyle='--')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Leaky Integrate-and-Fire Neuron")
plt.show()
