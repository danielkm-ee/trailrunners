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
    def __init__(self, device='cpu', ap=1, an=1, tp=2, tn=2, te=1, e_inc=0.5):
        super().__init__()
        self.tp = tp
        self.tn = tn
        self.te = te
        self.ap = ap
        self.an = an
        self.e_inc = e_inc
        self.init_state
        self.device = device

    def init_state(self):
        elig = torch.zeros(0)
        x = torch.zeros(0)
        y = torch.zeros(0)
        self.register_buffer("eligibility", elig, device=self.device)
        self.register_buffer("x", x, device=self.device)
        self.register_buffer("y", y, device=self.device)

    def update(self, pre, post):
        pre = pre.to(self.device)
        post = post.to(self.device)
        if len(pre) == len(post):
            num_steps = len(pre)
        else:
            num_steps = 0
            print("Error: Size of dim(0) of pre and post synaptic spikes should be equal. Got {} and {}".format(
                len(pre), len(post)))
        lay1 = pre.size()[1]
        lay2 = post.size()[1]

        self.x = torch.zeros(lay1, lay2, device=self.device)
        self.y = torch.zeros(lay1, lay2, device=self.device)
        self.eligibility = torch.zeros(lay1, lay2, device=self.device)
        for t in range(num_steps):
            self.x += (-self.x + self.ap*pre[t].repeat(lay2, 1).mT) / self.tp
            self.y += (-self.y + self.an*post[t].repeat(lay1, 1)) / self.tn

            self.eligibility += -self.eligibility / self.te + (self.x*post[t].repeat(lay1, 1) - self.y*pre[t].repeat(lay2, 1).mT)

        return self.eligibility

class SYNTDP(STDP):
    # synaptic connection layer with STDP-based weight updates
    def __init__(self, in_units, out_units, device='cpu', use_bias=True):
        super().__init__(device=device)
        self.weight = nn.Parameter(torch.randn(in_units, out_units, device=device, requires_grad=False))
        self.bias = nn.Parameter(torch.randn(out_units, device=device, requires_grad=False))
        self.use_bias = use_bias
        self.device = device

    def forward(self, input):
        return (input @ self.weight) + (self.bias if self.use_bias else torch.zeros_like(self.bias, device=self.device))

    def weight_update(self, pre, post, reward):
        self.weight.add_(reward * self.update(pre, post))
    
