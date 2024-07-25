# Performing the Trail task with a naive network
# Blame: Daniel Monahan
# contact: danielkm@github.com
#
# No training involved, this sample is primarily to test the runtime loop performance. Use this as a demo script for your experiments

import pygame
from simulator.TrailGame.game import Game, Controller
from simulator.TrailGame.environment import Map, Trail
from simulator.TrailGame.ant import Ant

import torch
import torch.nn as nn
import snntorch as snn

from snntorch import spikegen
from agents.lif_agent import LIF_SpikeNet
from agents.learning_rules.simple_Q import Simple_Q

import numpy as np
import time

# define variables and levels for trail game
SANTA_FE = [[0.0, 0.0], [20.0, 0.0], [40.0, 0.0], [60.0, 0.0], [60.0, 20.0], [60.0, 40.0], [60.0, 60.0], [60.0, 80.0],
        [60.0, 100.0], [80.0, 100.0], [100.0, 100.0], [120.0, 100.0], [160.0, 100.0], [180.0, 100.0], [200.0, 100.0],
        [220.0, 100.0], [240.0, 100.0], [240.0, 120.0], [240.0, 140.0], [240.0, 160.0], [240.0, 180.0], [240.0, 200.0],
        [240.0, 240.0], [240.0, 260.0], [240.0, 280.0], [240.0, 300.0], [240.0, 360.0], [240.0, 380.0], [240.0, 400.0],
        [240.0, 420.0], [240.0, 440.0], [240.0, 460.0], [220.0, 480.0], [200.0, 480.0], [180.0, 480.0], [160.0, 480.0],
        [140.0, 480.0], [100.0, 480.0], [80.0, 480.0], [40.0, 500.0], [40.0, 520.0], [40.0, 540.0], [40.0, 560.0],
        [60.0, 600.0], [80.0, 600.0], [100.0, 600.0], [120.0, 600.0], [160.0, 580.0], [160.0, 560.0], [180.0, 540.0],
        [200.0, 540.0], [220.0, 540.0], [240.0, 540.0], [260.0, 540.0], [280.0, 540.0], [300.0, 540.0], [340.0, 520.0],
        [340.0, 500.0], [340.0, 480.0], [340.0, 420.0], [340.0, 380.0], [340.0, 360.0], [340.0, 340.0], [360.0, 320.0],
        [420.0, 300.0], [420.0, 280.0], [420.0, 220.0], [420.0, 200.0], [420.0, 180.0], [420.0, 160.0], [440.0, 100.0],
        [460.0, 100.0], [500.0, 80.0], [500.0, 60.0], [520.0, 40.0], [540.0, 40.0], [560.0, 40.0], [580.0, 60.0],
        [580.0, 80.0], [580.0, 120.0], [580.0, 180.0], [580.0, 240.0], [560.0, 280.0], [540.0, 280.0], [520.0, 280.0],
        [460.0, 300.0], [480.0, 360.0], [540.0, 380.0], [520.0, 440.0], [460.0, 460.0]]
WIDTH = 640
HEIGHT = 640


def get_stimulus(state, intensity=0.7):
    '''
    Converts a boolean element of the game state into a 2-dim vector for use as an input to the network
    vector is normalized to magnitude 1, intensity defines the maximum value of the vector elements
    '''
    stimulus = torch.zeros(num_inputs)
    if state == True:
        stimulus[0] = intensity
        stimulus[1] = 1 - intensity
    else:
        stimulus[1] = intensity
        stimulus[0] = 1 - intensity
    return stimulus

def get_command(spikes):
    '''
    Uses the neuron with the most spikes as the output of the network
    '''
    command = ['l', 'r', 'f', 'n']
    _, idx = spikes.sum(dim=0).max(0)
    if _ <= 0:
        idx = 3 # no op

    if torch.rand(1) > 0.9: # random chance of moving
        idx = 2
    return command[idx]


# sim params
num_inputs = 2
num_hidden = 10
num_outputs = 3

num_steps = 200
num_moves = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float

# initialize the neural network
net = LIF_SpikeNet(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs, beta=0.5, num_steps=num_steps).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))

def main():
    pygame.init()

    # constructing game elements
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    ant = Ant();
    map_ = Map();
    trail = Trail();
    game = Game();

    for epoch in range(2):
        # load sf trail and update game state
        trail.load(SANTA_FE)
        game.update_state(screen, ant, map_, trail)
        game.update_score(ant, trail)

        movetime = []   # benchmarking runtime performance
        for move in range(num_moves):
            start = time.time()

            stimulus = get_stimulus(ant.smellsFood)
            stim_spk = spikegen.rate(stimulus, num_steps=num_steps, gain=1).to(device)
            spk1, mem1, spk2, mem2 = net(stim_spk)

            # decode outputs and play move
            command = get_command(spk2)
            game.play(ant, command, command=True)

            # update game state
            game.update_state(screen, ant, map_, trail)
            game.update_score(ant, trail)

            ## updating game state without drawing anything can be done using...
            # map_.patrol(ant)
            # ant.sniffAhead(trail)
            
            end = time.time() - start
            movetime.append(end)

            pygame.display.update()

        loss = torch.tensor(len(trail.pellets)) - torch.tensor(game.food_eaten)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    game.print_stats(ant, trail)
    print(f"Average time between moves: {np.mean(np.array(movetime))}")



if __name__ == '__main__':
    main()

