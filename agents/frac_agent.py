import snntorch as snn

import torch
import torch.nn as nn
import numpy as np
from agents.learning_rules.STDP_rec import Learner
import math


class FSNN(nn.Module):

        def __init__(self, num_input, num_hidden, num_output, num_steps, thinking_time, device):

                super().__init__()
                torch.set_grad_enabled(False)

                self.tottime = num_steps * thinking_time
                self.thinking_time = thinking_time

                
                self.forward_hidden = nn.Linear(num_input, num_hidden)
                self.forward_hidden.weight = nn.Parameter(torch.abs(self.forward_hidden.weight*10))
                hidden_weights = self.forward_hidden.weight
                

                self.feedback = nn.Linear(num_hidden, num_hidden)
                self.feedback.weight = nn.Parameter(torch.abs(self.feedback.weight*0))
                feedback_weights = self.feedback.weight
                
                
                self.hidden_layer = flif_neuron(num_hidden, device, self.tottime)

                
                self.forward_output = nn.Linear(num_hidden, num_output)
                self.forward_output.weight = nn.Parameter(torch.abs(self.forward_output.weight*30))
                output_weights = self.forward_output.weight

                
                self.output_layer = flif_neuron(num_output, device, self.tottime)


                self.forward_skip = nn.Linear(num_input, num_output)
                self.forward_skip.weight = nn.Parameter(torch.abs(self.forward_skip.weight*7.5))
                skip_weights = self.forward_skip.weight

                
                
                self.num_steps = num_steps
                self.device = device
                self.rescale = False
                self.rescale_count = 0

                # NOTE: Play with ratio of thinking time vs learner window, also the w_inc and max parameters
                w_inc_hid = math.sqrt(1 / num_input) * 0.25
                w_inc_out = math.sqrt(1 / num_hidden) * 0.25
                w_inc_fdb = math.sqrt(1 / num_hidden) * 0.25
                w_inc_skp = math.sqrt(1 / num_input) * 0.25

                w_max_hid = math.sqrt(1 / num_input) * 4
                w_max_out = math.sqrt(1 / num_hidden) * 10
                w_max_fdb = math.sqrt(1 / num_hidden) * 10
                w_max_skp = math.sqrt(1 / num_input) * 3

                window = int(thinking_time)
                self.learner = Learner(num_input, num_hidden, num_output, hidden_weights, output_weights, feedback_weights, skip_weights, window, w_inc_hid, w_inc_out, w_inc_fdb, w_inc_skp, w_max_hid, w_max_out, w_max_fdb, w_max_skp, device)
                

                self.hid_mem = self.hidden_layer.init_mem()
                self.out_mem = self.output_layer.init_mem()
                self.hid_spk = torch.zeros_like(self.hid_mem).to(device)
                self.out_spk = torch.zeros_like(self.out_mem).to(device)
                self.hist = False
                
        # Computes an action
        def forward(self, data):

                
                hidden_mem = self.hid_mem
                output_mem = self.out_mem

                hid_old_spk = self.hid_spk

                input_spikes = data

                hidden_current = self.forward_hidden(input_spikes)

                if not self.hist:
                        hidden_current.add_(self.feedback(self.hid_spk))

                self.hid_spk, self.hid_mem = self.hidden_layer(hidden_current, self.hid_mem)

                output_current = self.forward_output(self.hid_spk)

                output_current.add_(self.forward_skip(input_spikes))

                self.out_spk, self.out_mem = self.output_layer(output_current, self.out_mem)

                self.learner.update(input_spikes, self.hid_spk, hid_old_spk, self.out_spk, self.hist)

                self.hist = True

                return self.out_spk


        # Called after each action; 
        def weight_update(self, criticism):

                #if self.rescale:
                #        self.rescale_count += 1
                hidden_weights, output_weights, feedback_weights, skip_weights = self.learner.weight_change(criticism)


                self.forward_hidden.weight = nn.Parameter(hidden_weights)
                self.forward_output.weight = nn.Parameter(output_weights)
                self.feedback.weight = nn.Parameter(feedback_weights)
                self.forward_skip.weight = nn.Parameter(skip_weights)

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
                self.N = 0

        def update_delta(self, V_new, V_old):

                delta = torch.sub(V_new, V_old).detach()

                if self.N == 1:
                        # init delta_trace
                        self.delta_trace = torch.zeros(self.layer_size, self.num_steps).to(self.device)


                self.delta_trace[:, self.N-1] = delta
