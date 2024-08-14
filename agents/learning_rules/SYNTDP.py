# implementing STDP rule which allows for LTD and LTP
#
# 

### governing equations
# we use eligibility traces x and y, x for the presynaptic neuron and y for the postsynaptic neuron.
# the stdp agent will run through the output spikes for the network and create a vector of these eligibility traces
# which are updated according to (1)
#
# tp*x_dot = -x + a_positive(x)*spk
# tn*y_dot = -y + a_negative(y)*spk
#
# where a_positive(x) and a_negative(y) are the size of the weight increment for a spike, tp and tn are positive and negative
# decay rates, and spk is 1 if a neuron has spiked at time t
#
# 
import torch
import torch.nn as nn
import torch.nn.functional as F

class STDP(nn.Module):
    # tracks stdp updates of a connection matrix
    def __init__(self, ap=1, an=1, tp=2, tn=2, te=1, e_inc=0.5):
        super().__init__()
        self.tp = tp
        self.tn = tn
        self.te = te
        self.ap = ap
        self.an = an
        self.e_inc = e_inc

    def update(self, pre, post):
        if len(pre) == len(post):
            num_steps = len(pre)
        else:
            num_steps = 0
            print("Error: Size of dim(0) of pre and post synaptic spikes should be equal. Got {} and {}".format(
                len(pre), len(post)))

        x = torch.zeros_like(pre[1])
        y = torch.zeros_like(post[1])
        e = torch.zeros((pre.size()[1], post.size()[1]))
        for t in range(num_steps):
            xdot = (-x + self.ap*pre[t]) / self.tp
            x += xdot

            ydot = (-y + self.an*post[t]) / self.tn
            y += ydot

            for i in range(len(pre[1])):
                for j in range(len(post[1])):
                    edot = -e[i][j] / self.te + (x[i]*post[t][j] - y[j]*pre[t][i])
                    e[i][j] = e[i][j] + edot
        return e

class SYNTDP(STDP):
    # synaptic connection layer with STDP-based weight updates
    def __init__(self, in_units, out_units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, out_units, requires_grad=False))
        self.bias = nn.Parameter(torch.randn(out_units, requires_grad=False))

    def forward(self, input):
        return (input @ self.weight) + self.bias

    def weight_update(self, pre, post, reward):
        wdot = reward * self.update(pre, post)
        self.weight.add_(wdot)
    
