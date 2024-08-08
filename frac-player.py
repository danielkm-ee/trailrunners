

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
"""
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
"""
SANTA_FE = [[0.0, 0.0], [20.0, 0.0], [40.0, 0.0], [40.0, 20.0], [40.0, 40.0], [20.0, 40.0], [0.0, 40.0], [0.0, 20.0]]
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
        stimulus[1] = 0#1 - intensity
    else:
        stimulus[1] = intensity
        stimulus[0] = 0#1 - intensity
    return stimulus

def get_command(spikes):
    '''
    Uses the neuron with the most spikes as the output of the network
    '''
    global chosen_moves
    global random_moves
    
    command = ['f', 'l', 'r', 'n']
    _, idx = spikes.sum(dim=0).max(0)
    if _ <= 0:
        idx = 3 # no op
    else:
        chosen_moves += 1

    if torch.rand(1) > 0.99: # random chance of moving

        rand_move = torch.randint(low=0, high=3, size=(1,1))
        #idx = rand_move.item()
        #random_moves += 1

        #if _ > 0:
        #    chosen_moves -= 1
        
    return command[idx]

def dist_to_food(ant_pos, pellet_pos):

    delta_x = abs(ant_pos[0] - pellet_pos[0])

    if ant_pos[0] < pellet_pos[0]:

        toroid_x = ant_pos[0] + (WIDTH - pellet_pos[0])
    elif ant_pos[0] > pellet_pos[0]:

        toroid_x = pellet_pos[0] + (WIDTH - ant_pos[0])
    else:
        
        toroid_x = 0

    
    delta_y = abs(ant_pos[1] - pellet_pos[1])

    if ant_pos[1] < pellet_pos[1]:

        toroid_y = ant_pos[1] + (HEIGHT - pellet_pos[1])

    elif ant_pos[1] > pellet_pos[1]:

        toroid_y = pellet_pos[1] + (HEIGHT - ant_pos[1])

    else:

        toroid_y = 0

    dist = min(delta_x, toroid_x) + min(delta_y, toroid_y)
    
    return dist

# sim params
num_inputs = 2
num_hidden = 20
num_outputs = 3

num_steps = 200
num_moves = 10
num_epochs = 1000

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

    torch.set_grad_enabled(False)
    right_moves = 0
    high_score = 0

    tottime = num_steps * num_moves
    
    for epoch in range(num_epochs):

        
        
        trail = Trail()
        
        trail.load(SANTA_FE)
        game.update(ant, trail, map_)
        game.draw_screen(screen, ant, trail, map_)
        pygame.display.update()

        old_crit = 0
        
        net.reset()
        chosen_moves = 0
        random_moves = 0
        hunger = 0
        consec_food = 0
        noops = 0

        movetime = []   # benchmarking runtime performance
        #time.sleep(1)
        

        testing = False
        if right_moves > high_score:
            high_score = right_moves
            #go = input("Testing simulation. Execute?")
            #if go == 'y':
                #testing = True
            
        
        next_pellet = trail.pellets[0].position # update next pellet
        dist = dist_to_food(ant.position, next_pellet)

        
        for move in range(num_moves):
            pygame.event.pump()
            start = time.time()

            stimulus = get_stimulus(ant.sees_food_ahead)
            stim_spk = spikegen.rate(stimulus, num_steps=num_steps, gain=1).to(device)

            
            spike_train = list()
            for step in range(num_steps):

                step_spikes = net(stim_spk[step, :])

                spike_train.append(step_spikes)

            spike_train = torch.stack(spike_train)

            command = get_command(spike_train)
            if command == 'n':
                noops += 1
            game.play(ant, command, command=True)

            game.update(ant, trail, map_)
            game.draw_screen(screen, ant, trail, map_)
            pygame.display.update()
            
            # -----------------------------------------------
            
            if ant.was_fed:
                
                new_dist = dist_to_food(ant.position, next_pellet)
                inc = new_dist - dist
                
                if inc > 0:
                    criticism = -1

                else:

                    criticism = 1

                    if game.food_eaten == 8:
                        criticism = 100
                    else:
                        next_pellet = trail.pellets[0].position # update next pellet
                        dist = dist_to_food(ant.position, next_pellet)

                
            else:

                new_dist = dist_to_food(ant.position, next_pellet)
                inc = new_dist - dist


                dist = new_dist
                
                if inc > 0:

                    criticism = -1

                elif inc < 0:

                    criticism = 1

                else: # inc = 0
                    
                    criticism = 0
            # -----------------------------------------------
            end = time.time() - start
            #movetime.append(end)
            # ------------------------------------------------
            net.weight_update(criticism)

            if criticism < 0:
                right_moves = move + 1
                break
            

            
            
        #print_stats(ant, trail)
        print(f"Epoch: {epoch}")
        print(f"Moves made: {right_moves}. No-ops: {noops}. Food collected: {game.food_eaten}\n")


        game.reset(ant, map_, trail)
        game.draw_screen(screen, ant, trail, map_)
        pygame.display.update()


    # save parameters after training
    
    hid_con = net.forward_hidden.weight.detach().cpu().numpy()
    out_con = net.forward_output.weight.detach().cpu().numpy()
    fdb_con = net.feedback.weight.detach().cpu().numpy()
    skp_con = net.forward_skip.weight.detach().cpu().numpy()

    np.savez("../SFT_Training/parameters.npz", hid_con=hid_con, out_con=out_con, fdb_con=fdb_con, skp_con=skp_con)
    
    
    print("in-hid weights:", net.forward_hidden.weight)
    print("----------------------------------")
    print("hid-out weights:", net.forward_output.weight)
    print("----------------------------------")
    print("feedback weights:", net.feedback.weight)
    print("----------------------------------")
    print("skip weights:", net.forward_skip.weight)
    

if __name__ == '__main__':
    main()

