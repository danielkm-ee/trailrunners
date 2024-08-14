import pygame
from pygame.locals import *

from .environment import Trail

import os
from . import _utils

SPRITE_PATH = ''
SPRITE_PATH = "simulator/TrailGame/assets/agent2.png"#_utils.find_path(SPRITE_PATH, os.curdir, "agent.png")

SPRITE = pygame.image.load(SPRITE_PATH)
SPRITE = pygame.transform.scale(SPRITE, (20, 20)) # cell_size = 20

class Direction:
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class Ant:
    def __init__(self, position=[0, 0]):
        self.start_position = position.copy()
        self.position = position
        self.dir = Direction.RIGHT # 0 = right, 1 = up, 2 = left, 3 = down
        self.sees_food_ahead = False
        self.was_fed = False

    def move(self, cell_size=20):
        # when called, the ant will move 1 cell in the direction it is facing
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

    def checkCellAhead(self, trail, cell_size=20):
        # sets 'sees_food_ahead' to true if the ant is facing food, and wasFed to True if ant has eaten
        self.sees_food_ahead = False
        self.was_fed = False
        next_cell = self.position.copy()                                    # from ant.position [x, y]
        if (self.dir == Direction.RIGHT or self.dir == Direction.DOWN):
            next_cell[0 if self.dir == Direction.RIGHT else 1] += cell_size # get cell to right if dir==right, otherwise get cell below (dir==down)
        elif (self.dir == Direction.UP or self.dir == Direction.LEFT):
            next_cell[0 if self.dir == Direction.UP else 1] -= cell_size    # get cell to left.... get cell above...

        for pellet in trail.pellets:
            if (pellet.position == next_cell):
                self.sees_food_ahead = True
            if (pellet.position == self.position):
                self.was_fed = True

    def draw(self, surface):
        agent = pygame.transform.rotate(SPRITE, self.dir * (360/4))
        surface.blit(agent, (self.position[0], self.position[1]) )

