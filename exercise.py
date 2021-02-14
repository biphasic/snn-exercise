import numpy as np
import math
import torch
import torch.nn as nn
import ipdb


class Network:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, input_spikes):
        output_spikes = torch.zeros(
            (input_spikes.shape[0], self.layers[-1].output_dimension)
        )
        for i, spike_slice in enumerate(input_spikes):
            spikes = spike_slice
            for layer in self.layers:
                spikes = layer(spikes)
            output_spikes[i, :] = spikes
        return output_spikes


class LIFLayer(nn.Module):
    """
    A stateful Leaky Integrate and Fire (LIF) layer of neurons. As pyTorch does not explicitly support the solution of differential equations, we convert the ordinary differential equations that describe the neuron dynamics into difference equations and solve them at regular intervals dt.
    """

    def __init__(
        self, n_neurons, weights, voltage_tau=100, synapse_tau=50, spiking_threshold=1.0
    ):
        super(LIFLayer, self).__init__()
        self.weights = weights
        self.voltages = torch.zeros((1, n_neurons))
        self.voltage_decay = math.exp(-1 / voltage_tau)
        self.synapses = torch.zeros(weights.shape)
        self.synapse_decay = math.exp(-1 / synapse_tau)
        self.spiking_threshold = spiking_threshold
        self.output_dimension = weights.shape[1]
        self.voltage_traces = torch.empty((1, n_neurons))

    def forward(self, input_spikes):
        # decay voltages
        self.voltages *= self.voltage_decay

        # check if neurons spiked
        output_spikes = self.voltages >= self.spiking_threshold

        # decay synapse currents
        self.synapses = self.synapses * self.synapse_decay

        # increase synapse currents if input spiked
        for i in range(self.output_dimension):
            self.synapses[:, i] += self.weights[:, i] * input_spikes

        # integrate currents
        self.voltages += (
            self.synapses.sum(dim=0) - output_spikes * self.spiking_threshold
        )
        self.voltage_traces = torch.cat((self.voltage_traces, self.voltages), dim=0)
        return output_spikes
