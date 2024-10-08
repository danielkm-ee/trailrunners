import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple, deque
from itertools import count
import torch.optim as optim


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
n_actions = 3
n_observations = 2
optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

state = #TODO
memory = ReplayMemory(10000)


class DQN(nn.Module):

        def __init__(self, observations, actions):

                super(DQN, self).__init__()
                self.layer1 = nn.Linear(observations, 100)
                self.layer2 = nn.Linear(100, 100)
                self.layer3 == nn.Linear(100, actions)


        def forward(self, x):
                x = F.relu(self.layer1(x))
                x = F.relu(self.layer2(x))
                return self.layer3(x)
        
class ReplayMemory(object):

        def __init__(self, capacity):
                self.memory = deque([], maxlen=capacity)

        def push(self, *args):
                self.memory.append(Transition(*args))

        def sample(self, batch_size):
                return random.sample(self.memory, batch_size)

        def __len__(self):
                return len(self.memory)

def select_action(state, actions=3):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
                with torch.no_grad():
                        return policy_net(state).argmax(dim=0)
        else:
                return torch.tensor(random.sample(actions, 1))


def optimize_model():

        if len(memory) < 
