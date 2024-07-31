

import pygame
from simulator.TrailGame.game import Game, Controller
from simulator.TrailGame.environment import Map, Trail
from simulator.TrailGame.ant import Ant

import torch
import torch.nn as nn
import snntorch as snn

from snntorch import spikegen
from agents.frac_agent import FSNN

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

chosen_moves = 0
random_moves = 0

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
    global chosen_moves
    global random_moves
    
    command = ['l', 'f', 'r', 'n']
    _, idx = spikes.sum(dim=0).max(0)
    if _ <= 0:
        idx = 3 # no op
    else:
        chosen_moves += 1

    if torch.rand(1) > 0.95: # random chance of moving

        rand_move = torch.randint(low=0, high=3, size=(1,1))
        #idx = rand_move.item()
        #random_moves += 1

        #if _ > 0:
        #    chosen_moves -= 1
        
    return command[idx]


# sim params
num_inputs = 2
num_hidden = 20
num_outputs = 3

num_steps = 100
num_moves = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float

# initialize the neural network
net = FSNN(num_inputs, num_hidden, num_outputs, num_moves, num_steps, device).to(device)


def main():

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    ant = Ant()
    game = Game()
    map_ = Map()

    global chosen_moves
    global random_moves
    
    for epoch in range(100):

        
        
        trail = Trail()
        
        trail.load(SANTA_FE)
        game.update(ant, trail, map_)
        game.draw_screen(screen, ant, trail, map_)
        pygame.display.update()

        old_crit = 0
        
        net.reset()
        chosen_moves = 0
        random_moves = 0

        movetime = []   # benchmarking runtime performance
        time.sleep(1)
        for move in range(num_moves):
            start = time.time()

            stimulus = get_stimulus(ant.sees_food_ahead)
            stim_spk = spikegen.rate(stimulus, num_steps=num_steps, gain=1).to(device)
            spk2 = net(stim_spk)

            # decode outputs and play move
            command = get_command(spk2)
            game.play(ant, command, command=True)

            if ant.was_fed:
                criticism = 2

            elif ant.sees_food_ahead:
                criticism = 0.5

            else:
                criticism = -1

            #old_crit = critic.get_Q(game.food_eaten, num_moves)
                
            net.weight_update(criticism)

            # update game state
            game.update(ant, trail, map_)

            ## updating game state without drawing anything can be done using...
            # map_.patrol(ant)
            # ant.sniffAhead(trail)
            
            end = time.time() - start
            movetime.append(end)

            game.draw_screen(screen, ant, trail, map_)
            pygame.display.update()


        #print_stats(ant, trail)
        print(f"No ops: {net.scale}")
        print(f"Average time between moves: {np.mean(np.array(movetime))}")
        print(f"Random moves made: {random_moves}. Chosen moves made: {chosen_moves}")
        game.reset(ant, map_, trail)
        game.draw_screen(screen, ant, trail, map_)
        pygame.display.update()



if __name__ == '__main__':
    main()

