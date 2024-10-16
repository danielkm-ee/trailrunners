

import pygame
from simulator.TrailGame.game import Game, Controller
from simulator.TrailGame.environment import Map, Trail
from simulator.TrailGame.ant import Ant

import torch
import torch.nn as nn
import snntorch as snn

from snntorch import spikegen
from agents.lif_syntdp_net import SNN

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
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

PLOT_ON = False

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

    if torch.rand(1) > 0.99 and False: # random chance of moving

        rand_move = torch.randint(low=0, high=3, size=(1,1))
        idx = rand_move.item()
        random_moves += 1

        if _ > 0:
            chosen_moves -= 1
        
    return command[idx]


# sim params
num_inputs = 2
num_hidden = 20
num_outputs = 3

num_steps = 100
num_moves = 150

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
dtype = torch.float

torch.set_grad_enabled(False)

# initialize the neural network
net = SNN(num_inputs, num_hidden, num_outputs, num_steps, device=device).to(device)

matplotlib.rcParams['image.cmap'] = 'inferno'

best = []
current = []
def main():

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    ant = Ant()
    game = Game()
    map_ = Map()

    if PLOT_ON:
        plt.ion()
        gs1 = GridSpec(1, 3)
        fig1 = plt.figure()
        ax1_1 = fig1.add_subplot(gs1[0, 0])
        ax1_2 = fig1.add_subplot(gs1[0, 1])

        gs2 = GridSpec(2, 2)
        fig2 = plt.figure()
        ax2_1 = fig2.add_subplot(gs2[0, 0])
        ax2_2 = fig2.add_subplot(gs2[0, 1])
        ax2_3 = fig2.add_subplot(gs2[1, :])

    global chosen_moves
    global random_moves

    best_game_score = 0
    
    for epoch in range(100):

        trail = Trail()
        
        trail.load(SANTA_FE)
        game.update(ant, trail, map_)
        game.draw_screen(screen, ant, trail, map_)
        pygame.display.update()

        old_crit = 0
        
        chosen_moves = 0
        random_moves = 0

        movetime = []   # benchmarking runtime performance
        time.sleep(1)
        for move in range(num_moves):
            start = time.time()
            pygame.event.pump()

            stimulus = get_stimulus(ant.sees_food_ahead)
            stim_spk = spikegen.rate(stimulus, num_steps=num_steps, gain=1).to(device)
            spk1, mem1, spk2, mem2 = net(stim_spk)

            # decode outputs and play move
            command = get_command(spk2)
            game.play(ant, command, command=True)

            if ant.was_fed:
                criticism = 2

            elif ant.sees_food_ahead:
                criticism = 0.5

            else:
                criticism = -0.25

            #old_crit = critic.get_Q(game.food_eaten, num_moves)
                
            net.syn1.weight_update(stim_spk, spk1, criticism)
            net.syn2.weight_update(spk1, spk2, criticism)

            # update game state
            game.update(ant, trail, map_)

            ## updating game state without drawing anything can be done using...
            # map_.patrol(ant)
            # ant.sniffAhead(trail)
            
            end = time.time() - start
            movetime.append(end)

            if PLOT_ON:
                ax1_1.remove()
                ax1_2.remove()
                ax2_1.remove()
                ax2_2.remove()
                ax2_3.remove()
                ax1_1 = fig1.add_subplot(gs1[0, 0])
                ax1_2 = fig1.add_subplot(gs1[0, 1])
                ax1_3 = fig1.add_subplot(gs1[0, 2])
                ax2_1 = fig2.add_subplot(gs2[0, 0])
                ax2_2 = fig2.add_subplot(gs2[0, 1])
                ax2_3 = fig2.add_subplot(gs2[1, :])
                ax1_1.imshow(net.syn1.weight)
                ax1_2.imshow(net.syn2.weight)
                ax2_1.imshow(mem1.T)
                ax2_2.imshow(mem2.T)
                ax2_3.imshow(stim_spk.T)
                plt.pause(0.00000001)

            game.draw_screen(screen, ant, trail, map_)
            pygame.display.update()


        #print_stats(ant, trail)
        if game.food_eaten > best_game_score:
            best_game_score = game.food_eaten
        print(f"Best Game: {best_game_score}")
        print(f"This Game: {game.food_eaten}")
        print(f"No ops: ")
        print(f"Average time between moves: {np.mean(np.array(movetime))}")
        print(f"Random moves made: {random_moves}. Chosen moves made: {chosen_moves}")
        best.append(best_game_score)
        current.append(game.food_eaten)
        game.reset(ant, map_, trail)
        game.draw_screen(screen, ant, trail, map_)
        pygame.display.update()



if __name__ == '__main__':
    main()

