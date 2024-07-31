import pygame
from pygame.locals import *

class Map:
    # draws a grid and patrols borders, stopping or moving the ant when it reaches an edge
    def __init__(self, width=640, height=640, grid_size=32, background=pygame.Color(250, 245, 200), foreground=pygame.Color(111,111,111), toroidal=True):
        self.width = width
        self.grid_size = grid_size
        self.height = height
        self.background = background
        self.foreground = foreground
        self.toroidal = toroidal
    
    def checkBoundaries(self, ant): # prevents ant from exiting
        if ant.position[0] >= self.width:
            ant.position[0] = 0 if self.toroidal else self.width - self.width/self.grid_size

        if ant.position[0] < 0:
            ant.position[0] = self.width - self.width/self.grid_size if self.toroidal else 0

        if ant.position[1] >= self.height:
            ant.position[1] = 0 if self.toroidal else self.height - self.height/self.grid_size
        
        if ant.position[1] < 0:
            ant.position[1] = self.height - self.height/self.grid_size if self.toroidal else 0

    def draw(self, surface):
        surface.fill(self.background)
        
        for x in range(0, self.width, self.width // self.grid_size):
            for y in range(0, self.height, self.height // self.grid_size):
                pygame.draw.line(surface, self.foreground, (x, 0), (x, self.height), 2)
                pygame.draw.line(surface, self.foreground, (0, y), (self.width, y), 2)


class Pellet:
    def __init__(self, position=[20 * 16, 20 * 16], color=pygame.Color(125, 10, 10)):
        self.position = position
        self.color = color

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, pygame.Rect(self.position[0], self.position[1], 20, 20))

class Trail:
    def __init__(self):
        self.pellets = list()
        self.total_pellets = 0
    
    def addPellet(self, position):
        pellet = Pellet(position=position)
        self.pellets.append(pellet)
        self.total_pellets += 1

    def load(self, positions):
        for pos in positions:
                self.addPellet(pos)

    def update(self):
        for i in range(len(self.pellets)):
            if self.pellets[i-1].position == ant.position:
                self.pellets.pop[i-1]

    def removePellet(self, pelletIndex):
            self.pellets.pop(pelletIndex)

    def draw(self, surface):
        for i in range(len(self.pellets)):
            self.pellets[i].draw(surface)
