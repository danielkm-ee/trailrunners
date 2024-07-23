# Simulation environment for Neural networks performing foraging tasks
# Author(s) : Daniel Monahan
# contact(s) : danielkm@github.com
#
# 

import pygame
from pygame.locals import *

from ant import Ant
from environment import Map, Trail

class Controller:
    def rlm_controls(ant, key):
        if (key == pygame.K_j):
            ant.dir = (ant.dir + 1) % 4
        elif (key == pygame.K_k):
            ant.dir = (ant.dir - 1) % 4
        elif (key == pygame.K_f):
            ant.move()
        return

    def rlm_command(ant, command):
        if (command == 'lt'):
            ant.dir = (ant.dir + 1) % 4
        elif (key == 'rt'):
            ant.dir = (ant.dir - 1) % 4
        elif (key == 'f'):
            ant.move()
        return

    def pellet_on_click(trail, cell_size=20):
        mousePosition = pygame.mouse.get_pos()
        xPos = mousePosition[0] - mousePosition[0] % (cell_size);
        yPos = mousePosition[1] - mousePosition[1] % (cell_size);
        trail.addPellet([xPos, yPos]);

class Game:
    def __init__(self):
        self.food_eaten = 0
        self.steps_taken = 0
        self.score = 0

    def play(self, ant, move, command=False):
        if command:
            Controller.rlm_command(ant, move)
        else:
            Controller.rlm_controls(ant, move)
        self.steps_taken += 1

    def update_state(self, screen, ant, map_, trail):
        map_.guard(ant)
        ant.sniffAhead(trail)
        map_.draw(screen)
        trail.draw(screen)
        ant.draw(screen)

    def update_score(self, ant, trail):
        ant.wasFed = False
        for idx in range(len(trail.pellets)):
            if trail.pellets[idx-1].position == ant.position:
                self.food_eaten += 1
                trail.removePellet(idx-1)
                ant.wasFed = True
        self.score = self.food_eaten

