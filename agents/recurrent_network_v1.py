# Recurrent LIF Agent
# contact : danielkm@github.com
#
# Implementation of a recurrent spiking network using LSTM and GRU inspired recurrent connections
#

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from agents.learning_rules.STDP_rec import Learner

class RSNN_LSTM(nn.Module):
    '''
    spiking network with recurrent connections, a total beast of a network ToT
    '''
    def __init__(self, num_inputs, num_hidden, num_outputs, num_steps, device, beta=0.5, alpha=0.5):
        super().__init__()
        
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.rc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = LIFv1()
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = LIF()
        self.num_steps = num_steps
        self.rescale = False
        self.rescale_count = 0

        hidden_weights = self.fc1.weight
        feedback_weights = self.rc1.weight
        output_weights = self.fc2.weight

        self.learner = Learner(num_inputs, num_hidden, num_outputs, hidden_weights, output_weights, feedback_weights, int(num_steps / 4), 0.1, 0.1, 0.1, 20, device)


    def forward(self, x):

        # initalize hidden states
        self.lif1.reset_mem()
        self.lif2.reset_mem()

        mem1_rec = []
        spk1_rec = []
        mem2_rec = []
        spk2_rec = []

        for step in range(self.num_steps):
            candidate1 = F.sigmoid(self.fc1(x[step]))
            forget1 = F.relu(self.rc1(x[step]))
            spk1, mem1 = self.lif1(candidate1, forget1)

            syn2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(syn2)
            mem1_rec.append(mem1)
            mem2_rec.append(mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

            self.learner.update(x[step], spk1, spk2, do_feedback=True)


        spk1_rec = torch.stack(spk1_rec, dim=0)
        mem1_rec = torch.stack(mem1_rec, dim=0)
        spk2_rec = torch.stack(spk2_rec, dim=0)
        mem2_rec = torch.stack(mem2_rec, dim=0)

        spike_count = spk2_rec.sum().item()
        self.rescale = (spike_count == 0)

        return spk1_rec, mem1_rec, spk2_rec, mem2_rec

    # Called after each action 
    def weight_update(self, criticism):

            if self.rescale:
                    self.rescale_count += 1
            hidden_weights, output_weights, feedback_weights = self.learner.weight_change(criticism, self.rescale)

            self.fc1.weight = nn.Parameter(hidden_weights)
            self.fc2.weight = nn.Parameter(output_weights)
            self.rc1.weight = nn.Parameter(feedback_weights)


class LIF(nn.Module):
    '''
    Recurrent Synaptic Leaky integrate and fire neuron with surrogate gradient
    '''
    def __init__(self, threshold=1, alpha=0.7, beta=0.5):
        super(LIF, self).__init__()
        self.beta = beta                # membrane leak
        self.alpha = alpha              # synaptic leak
        self.threshold = threshold
        self.spike_fn = self.ATan.apply
        self.init_mem()
        self.init_syn()

    def forward(self, x):
        # computes membrane potential and spike ocurrence
        if not self.mem.shape == x.shape:
            self.mem = torch.zeros_like(x)
        if not self.syn.shape == x.shape:
            self.syn = torch.zeros_like(x)

        spk = self.spike_fn(self.mem-self.threshold)    # out spk
        self.syn = self.beta*self.syn + x               # synaptic current
        self.mem = self.beta*self.mem + self.syn - self.threshold*spk # membrane potential

        return spk, self.mem

    def init_mem(self):
        # initialize membrane potential
        mem = torch.zeros(0)
        self.register_buffer("mem", mem, persistent=False)

    def init_syn(self):
        # initialize synaptic current
        syn = torch.zeros(0)
        self.register_buffer("syn", syn, persistent=False)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.mem
    
    def reset_syn(self):
        self.syn = torch.zeros_like(self.mem, device=self.mem.device)
        return self.syn

    class ATan(torch.autograd.Function):
        # heaviside spike function with fast sigmoid surrogate gradient
        @staticmethod
        def forward(ctx, mem):
            spk = (mem > 0).float()
            ctx.save_for_backward(mem)
            return spk

        @staticmethod
        def backward(ctx, grad_output):
            (mem,) = ctx.saved_tensors
            grad = 1 / (1 + (np.pi*mem).pow_(2)) * grad_output
            return grad

class LIFv1(nn.Module):
    '''
    ~Improved (v1)~ Recurrent Synaptic Leaky integrate and fire neuron with surrogate gradient
    note: has modified synaptic update using LSTM/GRU like recurrent connections
    '''
    def __init__(self, threshold=1, alpha=0.7, beta=0.5):
        super().__init__()
        self.beta = beta                # membrane leak
        self.threshold = threshold
        self.spike_fn = self.ATan.apply
        self.init_mem()
        self.init_syn()

    def forward(self, candidate, forget):
        # computes membrane potential and spike ocurrence
        if not self.mem.shape == candidate.shape:
            self.mem = torch.zeros_like(candidate)
        if not self.syn.shape == candidate.shape:
            self.syn = torch.zeros_like(candidate)

        self.syn = forget*self.syn + (1 - forget) * candidate       # synaptic current update
        spk = self.spike_fn(self.mem-self.threshold)    # out spk
        self.mem = self.beta*self.mem + self.syn - self.threshold*spk # membrane potential

        return spk, self.mem

    def init_mem(self):
        # initialize membrane potential to empty tensor
        mem = torch.zeros(0)
        self.register_buffer("mem", mem, persistent=False)

    def init_syn(self):
        # initialize synaptic current to empty tensor
        syn = torch.zeros(0)
        self.register_buffer("syn", syn, persistent=False)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.mem
    
    def reset_syn(self):
        self.syn = torch.zeros_like(self.mem, device=self.mem.device)
        return self.syn

    class ATan(torch.autograd.Function):
        # heaviside spike function with fast sigmoid surrogate gradient
        @staticmethod
        def forward(ctx, mem):
            spk = (mem > 0).float()
            ctx.save_for_backward(mem)
            return spk

        @staticmethod
        def backward(ctx, grad_output):
            (mem,) = ctx.saved_tensors
            grad = 1 / (1 + (np.pi*mem).pow_(2)) * grad_output
            return grad
