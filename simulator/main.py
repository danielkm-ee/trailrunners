# Simulation environment for Neural networks performing foraging tasks
# Author(s) : Daniel Monahan
# contact(s) : danielkm@github.com
#
# This file demonstrates the runtime loop for each frame of the trailrunning game. This file is a playable version, using keys to control the character.
# The runtime loop is the same if using an artificial agent, but use the command=True tag when calling 'play' as this will allow chars in {'r', 'l', 'f'}
# to be sent as controls (commands) for the gameplay.

import pygame
import time
import os

from game import Game, Controller
from environment import Map, Trail
from ant import Ant

import _utils

GRID_SIZE = 32

# window size
SCALE = 20
WIDTH = GRID_SIZE * SCALE
HEIGHT = GRID_SIZE * SCALE

TOROIDAL_MAP = True     # if true edges, are wrapped around to eachother. otherwise edges are walls

# loading trails into RAM
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

# colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
gray = pygame.Color(150, 150, 150)
red = pygame.Color(125, 10, 10)
lightyellow = pygame.Color(250, 245, 200)


def main():
    # Initialising pygame
    pygame.init()

    # constructing game elements
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    ant = Ant();
    map_ = Map();
    trail = Trail();
    game = Game();
    # font = pygame.font.Font

    while True:

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                game.play(ant, event.key, command=False)

                if event.key == pygame.K_s: # load santa fe trail
                    trail.load(SANTA_FE)

                if event.key == pygame.K_q:
                    pygame.quit()
                    game.print_stats(ant)
                    quit()


            if event.type == pygame.MOUSEBUTTONDOWN:
                Controller.pellet_on_click(trail, WIDTH//GRID_SIZE)

            if event.type == pygame.QUIT:
                pygame.quit()
                game.print_stats(ant)
                quit()
            
        game.update_state(screen, ant, map_, trail)
        game.update_score(ant, trail)

        ## manual version of game.update()
        # map_.patrol(ant)
        # map_.draw(screen)
        # trail.draw(screen)
        # ant.draw(screen)
        # trail.update()

        pygame.display.update()



if __name__ == '__main__':
    main()
