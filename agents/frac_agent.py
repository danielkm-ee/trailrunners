import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen



import torch
import torch.nn as nn


class FSNN(nn.Module):

        def __init__(self, num_input, num_hidden, num_output, num_steps, device):

                super().__init__()

                self.forward_hidden = nn.Linear(num_input, num_hidden)
                self.hidden_layer = flif_neuron(num_hidden, device, num_steps)

                self.forward_output = nn.Linear(num_hidden, num_output)
                self.output_layer = flif_neuron(num_output, device, num_steps)

                self.num_steps = num_steps
                self.device = device

        def forward(self, data):

                # TODO: fill



class flif_neuron(nn.Module):

        weight_vector = list()

        def __init__(self, size, device, num_steps):

                super().__init__()

                self.layer_size = size
                self.device = device
                self.num_steps = num_steps
                self.delta_trace = torch.zeros(0)
                self.spike_gradient = snn.surrogate.ATan
                
                # Fractional LIF equation parameters
                self.alpha = 0.2
                self.dt = 0.1
                self.threshold = -50
                self.V_init = -70
                self.VL = -70
                self.V_reset = -70
                self.gl = 0.025
                self.Cm = 0.5


                if len(FLIF.weight_vector) == 0:
                        x = num_steps
                        
                        nv = np.arange(x-1)
                        flif_neuron.weight_vector = torch.tensor((x+1-nv)**(1-self.alpha)-(x-nv)**(1-self.alpha)).float().to(self.device)

        
