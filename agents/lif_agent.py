# Spiking Neural Network trailrunner
# Author : Daniel Monahan
# danielkm@github.com
#
# This is a simple recurrent spiking neural network which can be trained to solve trail-running games such as 
# the Santa Fe Trail

import torch
import torch.nn as nn
import snntorch as snn
from agents.learning_rules.STDP import Learner

class SNN(nn.Module):
    def __init__(self, num_inputs=2, num_hidden=10, num_outputs=4, num_steps=100, beta=0.5, device='cpu'):
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        hidden_weights = self.fc1.weight
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        output_weights = self.fc2.weight
        self.lif2 = snn.Leaky(beta=beta, output=True)

        self.num_steps = num_steps
        self.learner = Learner(num_inputs, num_hidden, num_outputs, hidden_weights, output_weights, 
                window=50, w_inc_hid=0.01, w_inc_out=0.01, w_s_max=10, device=device)

        self.rescale = False
        self.rescale_count = 0

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
            syn1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(syn1, mem1)
            syn2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(syn2, mem2)

            mem1_rec.append(mem1) # record outputs
            mem2_rec.append(mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

            # update STDP mechanism
            self.learner.update(x[step], spk1, spk2)

        # convert spikes to tensors for future operations
        spk1_rec = torch.stack(spk1_rec, dim=0)
        mem1_rec = torch.stack(mem1_rec, dim=0)
        spk2_rec = torch.stack(spk2_rec, dim=0)
        mem2_rec = torch.stack(mem2_rec, dim=0)

        self.rescale = (spk2_rec.sum().item() == 0) # if no spikes, rescale = True
        

        return spk1_rec, mem1_rec, spk2_rec, mem2_rec

    # Called after each action; 
    def weight_update(self, criticism):

            self.rescale_count += 1
            hidden_weights, output_weights = self.learner.weight_change(criticism, self.rescale)

            self.fc1.weight = nn.Parameter(hidden_weights)
            self.fc2.weight = nn.Parameter(output_weights)

