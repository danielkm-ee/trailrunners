# Simulation environment for Neural networks performing foraging tasks
# Author(s) : Daniel Monahan
# contact(s) : danielkm@github.com
#
# 

import pygame
from pygame.locals import *

from .ant import Ant
from .environment import Map, Trail

class Controller:
    # interprets commands and controls from the user/agent
    def rlm_controls(ant, key):
        # j = left turn, k = right turn, f = forward
        if (key == pygame.K_j):
            ant.dir = (ant.dir + 1) % 4
        elif (key == pygame.K_k):
            ant.dir = (ant.dir - 1) % 4
        elif (key == pygame.K_f):
            ant.move()
        return

    def rlm_command(ant, command):
        if (command == 'l'):
            ant.dir = (ant.dir + 1) % 4
        elif (key == 'r'):
            ant.dir = (ant.dir - 1) % 4
        elif (key == 'f'):
            ant.move()
        return

    def pellet_on_click(trail, cell_size=20):
        # places a pellet at mouse position
        mousePosition = pygame.mouse.get_pos()
        xPos = mousePosition[0] - mousePosition[0] % (cell_size);
        yPos = mousePosition[1] - mousePosition[1] % (cell_size);
        for pellet in trail.pellets:
            if (pellet.position == [xPos, yPos]):
                return
        trail.addPellet([xPos, yPos])
        return

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
        map_.patrol(ant)
        ant.sniffAhead(trail)
        map_.draw(screen)
        trail.draw(screen)
        ant.draw(screen)

    def update_score(self, ant, trail):
        for idx in range(len(trail.pellets)):
            if trail.pellets[idx-1].position == ant.position:
                self.food_eaten += 1
                trail.removePellet(idx-1)
        self.score = self.food_eaten

    def print_stats(self, ant, trail):
        print("PERFORMANCE STATISTICS")
        print(f"Score: {self.score}")
        print(f"Food Eaten: {self.food_eaten}")
        print(f"Steps Taken: {self.steps_taken}")
        print(f"Total Food: {trail.total_pellets}")
        print(f"Ant was fed?: {ant.wasFed}")
        print(f"Ant smells food?: {ant.smellsFood}")

