import snntorch as snn

import torch
import torch.nn as nn
import numpy as np
from agents.learning_rules.STDP import Learner
import math


class FSNN(nn.Module):

        def __init__(self, num_input, num_hidden, num_output, num_steps, thinking_time, device):

                super().__init__()

                self.tottime = num_steps * thinking_time
                self.thinking_time = thinking_time

                self.forward_hidden = nn.Linear(num_input, num_hidden)
                hidden_weights = self.forward_hidden.weight
                #self.forward_hidden.weight = nn.Parameter(torch.abs(weight.mul(5)))


                #self.recurrent_hidden = nn.Linear(num_hidden, num_hidden)
                #weight = self.recurrent_hidden.weight
                #self.forward_hidden.weight = nn.Parameter(torch.abs(weight.mul(5)))
                
                
                self.hidden_layer = flif_neuron(num_hidden, device, self.tottime)

                self.forward_output = nn.Linear(num_hidden, num_output)
                output_weights = self.forward_output.weight
                #self.forward_output.weight = nn.Parameter(torch.abs(weight.mul(5)))
                
                self.output_layer = flif_neuron(num_output, device, self.tottime)

                self.num_steps = num_steps
                self.device = device

                self.rescale = False
                self.rescale_count = 0

                # NOTE: Play with ratio of thinking time vs learner window, also the w_inc and max parameters
                self.learner = Learner(num_input, num_hidden, num_output, hidden_weights, output_weights 50, 0.1, 0.1, 10, device)
                
                
                
        # Computes an action
        def forward(self, data):

                hidden_mem = self.hidden_layer.init_mem()
                output_mem = self.output_layer.init_mem()

                spike_trace = list()
                

                # TODO: fill
                # feed input to network, come up with an action (spike train)
                for ms in range(self.thinking_time):

                        # forward pass
                        # data should be thinking_time x num_input
                        input_spikes = data[ms, :]
                        
                        hidden_current = self.forward_hidden(input_spikes)

                        #if ms > 0:

                                #hidden_current.add_(self.recurrent_hidden(hidden_spikes))
                        
                        hidden_spikes, hidden_mem = self.hidden_layer(hidden_current, hidden_mem)
                        
                        output_current = self.forward_output(hidden_spikes)
                        output_spikes, output_mem = self.output_layer(output_current, output_mem)

                        spike_trace.append(output_spikes)

                        self.learner.update(input_spikes, hidden_spikes, output_spikes)


                spike_trace = torch.stack(spike_trace, dim=0)

                spike_count = spike_trace.sum().item()

                self.rescale = (spike_count == 0)
                
                return spike_trace

        # Called after each action; 
        def weight_update(self, criticism):

                self.rescale_count += 1
                hidden_weights, output_weights = self.learner.weight_change(criticism, self.rescale)


                self.forward_hidden.weight = nn.Parameter(hidden_weights)
                self.forward_output.weight = nn.Parameter(output_weights)


        def reset(self):
                self.rescale_count = 0
                self.hidden_layer.reset_memory()
                self.output_layer.reset_memory()



class flif_neuron(nn.Module):

        weight_vector = list()

        def __init__(self, size, device, num_steps):

                super().__init__()

                self.layer_size = size
                self.device = device
                self.num_steps = num_steps
                self.delta_trace = torch.zeros(0)
                
                # Fractional LIF equation parameters
                self.alpha = 0.2
                self.dt = 1 #ms
                self.threshold = -50
                self.V_init = -70
                self.VL = -70
                self.V_reset = -70
                self.gl = 0.025
                self.Cm = 0.5
                self.N = 0


                if len(flif_neuron.weight_vector) == 0:
                        x = num_steps
                        
                        nv = np.arange(x-1)
                        flif_neuron.weight_vector = torch.tensor((x+1-nv)**(1-self.alpha)-(x-nv)**(1-self.alpha)).float().to(self.device)


        def forward(self, I, V_old):

                if self.N == 0:

                        V_new = (torch.ones_like(V_old)*self.V_init).to(self.device)
                        spike = torch.zeros_like(V_old).to(self.device)
                        self.N += 1

                        return spike, V_new

                elif self.N == 1:
                        # Classical LIF
                        tau = self.Cm / self.gl
                        V_new = V_old + (self.dt/tau)*(-1 * (V_old - self.VL) + I/self.gl)

                else:
                        # Fractional LIF
                        V_new = self.dt**(self.alpha) * math.gamma(2-self.alpha) * (-self.gl*(V_old-self.VL)+I) / self.Cm + V_old

                        delta_trace = self.delta_trace[:, 0:self.N-1]

                        weights = flif_neuron.weight_vector[-self.N+1:]
                        memory_V = torch.matmul(delta_trace, weights)

                        V_new = torch.sub(V_new, memory_V)


                spike = ((V_old - self.threshold) > 0).float()
                reset = (spike * (V_new - self.V_reset)).detach()

                V_new = torch.sub(V_new, reset)
                self.update_delta(V_new, V_old)
                self.N += 1

                return spike, V_new

        def init_mem(self):

                #self.delta_trace = torch.zeros(0)

                self.N = 0
                return torch.zeros(self.layer_size).to(self.device)

        def reset_memory(self):

                self.delta_trace = torch.zeros(0).to(self.device)

        def update_delta(self, V_new, V_old):

                delta = torch.sub(V_new, V_old).detach()

                if self.N == 1:
                        # init delta_trace
                        self.delta_trace = torch.zeros(self.layer_size, self.num_steps).to(self.device)


                self.delta_trace[:, self.N-1] = delta
