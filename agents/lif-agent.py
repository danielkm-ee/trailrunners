# Spiking Neural Network trailrunner
# Author : Daniel Monahan
# danielkm@github.com
#
# This is a simple recurrent spiking neural network which can be trained to solve trail-running games such as 
# the Santa Fe Trail

import torch
import torch.nn as nn
import snntorch as snn

class Net(nn.Module):
    def __init__(self):
        super().__init__(num_inputs=2, num_hidden=10, num_outputs=4, beta=0.5)

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, output=True)

    def forward(self, x):

        # initalize hidden states
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()

        mem1_rec = []
        spk1_rec = []
        mem2_rec = []
        spk2_rec = []

        for step in range(num_steps):
            syn1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(syn1, mem1)
            syn2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(syn2, mem2)
            mem1_rec.append(mem1)
            mem2_rec.append(mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

        spk1_rec = torch.stack(spk1_rec, dim=0)
        mem1_rec = torch.stack(mem1_rec, dim=0)
        spk2_rec = torch.stack(spk2_rec, dim=0)
        mem2_rec = torch.stack(mem2_rec, dim=0)

        return spk1_rec, mem1_rec, spk2_rec, mem2_rec


