# Spiking Neural Network trailrunner
# Author : Daniel Monahan
# danielkm@github.com
#
# This is a simple recurrent spiking neural network which can be trained to solve trail-running games such as 
# the Santa Fe Trail

import torch
import torch.nn as nn
import snntorch as snn
from agents.learning_rules.SYNTDP import SYNTDP

class SNN(nn.Module):
    def __init__(self, num_inputs=2, num_hidden=10, num_outputs=4, num_steps=100, beta=0.5, device='cpu'):
        super().__init__()

        self.syn1 = SYNTDP(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.syn2 = SYNTDP(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, output=True)

        self.num_steps = num_steps

    def forward(self, x):

        # initalize hidden states
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()

        # recording all layers for analysis, spk2 is output
        mem1_rec = []
        spk1_rec = []
        mem2_rec = []
        spk2_rec = []

        for step in range(self.num_steps):
            # propagate spikes through network
            curr1 = self.syn1(x[step])
            spk1, mem1 = self.lif1(curr1, mem1)
            curr2 = self.syn2(spk1)
            spk2, mem2 = self.lif2(curr2, mem2)

            mem1_rec.append(mem1) # record outputs
            mem2_rec.append(mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

        # convert spikes to tensors for future operations
        spk1_rec = torch.stack(spk1_rec, dim=0)
        mem1_rec = torch.stack(mem1_rec, dim=0)
        spk2_rec = torch.stack(spk2_rec, dim=0)
        mem2_rec = torch.stack(mem2_rec, dim=0)

        return spk1_rec, mem1_rec, spk2_rec, mem2_rec
