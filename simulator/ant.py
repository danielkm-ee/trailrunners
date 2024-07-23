import pygame
from pygame.locals import *

from environment import Trail

import os
import _utils

SPRITE_PATH = ''
SPRITE_PATH = _utils.find_path(SPRITE_PATH, os.curdir, "agent.png")

SPRITE = pygame.image.load(
        SPRITE_PATH)
SPRITE = pygame.transform.scale(SPRITE, (20, 20))

class Direction:
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class Ant:
    def __init__(self, position=[0, 0]):
        self.position = position
        self.dir = Direction.RIGHT # 0 = right, 1 = up, 2 = left, 3 = down
        self.smellsFood = False
        self.wasFed = False

    def move(self, cell_size=20):
        if (self.dir == Direction.RIGHT):
            self.position[0] += cell_size
        elif (self.dir == Direction.UP):
            self.position[1] -= cell_size
        elif (self.dir == Direction.LEFT):
            self.position[0] -= cell_size
        elif (self.dir == Direction.DOWN):
            self.position[1] += cell_size
        else :
            raise ValueError
            print("Ant.dir is out of range!")

    def sniffAhead(self, trail, cell_size=20):
        # sets 'smellsFood' to true if the ant is facing food
        self.smellsFood = False
        next_cell = self.position.copy()
        if (self.dir == 0 or self.dir == 3):
            next_cell[0 if self.dir == 0 else 1] += cell_size
        elif (self.dir == 1 or self.dir == 2):
            next_cell[0 if self.dir == 1 else 1] -= cell_size

        for i in range(len(trail.pellets)):
            if (trail.pellets[i-1].position == next_cell):
                self.smellsFood = True

    def draw(self, surface):
        agent = pygame.transform.rotate(SPRITE, self.dir * (360/4))
        surface.blit(agent, (self.position[0], self.position[1]) )

