import numpy as np

class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron model
    """

    def __init__(
        self,
        tau_m=20.0, # membrane time constant (ms)
        v_rest=-65.0, # resting potential (mV)
        v_reset=-65.0, # reset potential after spike (mV)
        v_threshold=-50.0, # spike threshold (mV)
        r_m=1.0, # membrane resistance
        dt=1.0 # simulation time step (ms)
    ):
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.r_m = r_m
        self.dt = dt

        self.v = v_rest # membrane potential
        self.spike = False

    def step(self, input_current):
        """
        Update neuron state for one time step
        """
        dv = (
            (-(self.v - self.v_rest) + self.r_m * input_current)
            / self.tau_m
        ) * self.dt

        self.v += dv

        if self.v >= self.v_threshold:
            self.spike = True
            self.v = self.v_reset
        else:
            self.spike = False

        return self.v, self.spike
