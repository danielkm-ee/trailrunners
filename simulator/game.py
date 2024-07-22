# Simulation environment for Neural networks performing foraging tasks
# Author(s) : Daniel Monahan
# contact(s) : danielkm@github.com
#
# 


# WIP Currently porting controls, object classes, and trail functionality from trail-game
import pygame
import time
import os

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

SPRITE_PATH = ''
SPRITE_PATH = _utils.find_path(SPRITE_PATH, os.curdir, "agent.png")

SPRITE = pygame.image.load(
        SPRITE_PATH)
SPRITE = pygame.transform.scale(SPRITE, (WIDTH//GRID_SIZE, HEIGHT//GRID_SIZE))

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

    def move(self):
        if (self.dir == Direction.RIGHT):
            self.position[0] += WIDTH // GRID_SIZE
        elif (self.dir == Direction.UP):
            self.position[1] -= HEIGHT // GRID_SIZE
        elif (self.dir == Direction.LEFT):
            self.position[0] -= WIDTH // GRID_SIZE
        elif (self.dir == Direction.DOWN):
            self.position[1] += HEIGHT // GRID_SIZE
        else :
            raise ValueError
            print("Ant.dir is out of range!")

    def sniffAhead(self, trail):
        # returns true if the ant is facing food
        self.smellsFood = False
        next_cell = self.position.copy()
        if (self.dir == 0 or self.dir == 3):
            next_cell[0 if self.dir == 0 else 1] += WIDTH // GRID_SIZE
        elif (self.dir == 1 or self.dir == 2):
            next_cell[0 if self.dir == 1 else 1] -= WIDTH // GRID_SIZE

        for i in range(len(trail.pellets)):
            if (trail.pellets[i-1].position == next_cell):
                self.smellsFood = True

    def draw(self, surface):
        agent = pygame.transform.rotate(SPRITE, self.dir * (360/4))
        surface.blit(agent, (self.position[0], self.position[1]) )


class Map:
    def __init__(self, width=WIDTH, height=HEIGHT, color=lightyellow, toroidal=TOROIDAL_MAP):
        self.width = width
        self.height = height
        self.color = color
        self.toroidal = toroidal
    
    def guard(self, ant): # prevents ant from exiting
        if ant.position[0] >= self.width:
            ant.position[0] = 0 if self.toroidal else self.width - self.width/GRID_SIZE

        if ant.position[0] < 0:
            ant.position[0] = self.width - self.width/GRID_SIZE if self.toroidal else 0

        if ant.position[1] >= self.height:
            ant.position[1] = 0 if self.toroidal else self.height - self.height/GRID_SIZE
        
        if ant.position[1] < 0:
            ant.position[1] = self.height - self.height/GRID_SIZE if self.toroidal else 0

    def draw(self, surface):
        surface.fill(self.color)
        
        for x in range(0, self.width, self.width // GRID_SIZE):
            for y in range(0, self.height, self.height // GRID_SIZE):
                pygame.draw.line(surface, gray, (x, 0), (x, self.height), 2)
                pygame.draw.line(surface, gray, (0, y), (self.width, y), 2)


class Pellet:
    def __init__(self, position=[WIDTH/GRID_SIZE * 16, HEIGHT/GRID_SIZE * 16]):
        self.position = position
        self.wasEaten = False

    def draw(self, surface):
        pygame.draw.rect(surface, red, pygame.Rect(self.position[0], self.position[1], 
                                                        WIDTH / GRID_SIZE, HEIGHT / GRID_SIZE))

class Trail:
    def __init__(self):
        self.pellets = list()
    
    def addPellet(self, position):
        pellet = Pellet(position)
        self.pellets.append(pellet)

    def load(self, trailname):
        if trailname == "santa fe":
            for pos in SANTA_FE:
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


# Initialising pygame
pygame.init()

# FPS (frames per second) controller
fps = 120



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

    def pellet_on_click():
        mousePosition = pygame.mouse.get_pos()
        xPos = mousePosition[0] - mousePosition[0] % (WIDTH / GRID_SIZE);
        yPos = mousePosition[1] - mousePosition[1] % (HEIGHT / GRID_SIZE);
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

font = pygame.font.Font
screen = pygame.display.set_mode((WIDTH, HEIGHT))
ant = Ant();
map_ = Map();
trail = Trail();
game = Game();

while True:

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            game.play(ant, event.key)

            if event.key == pygame.K_s: # load santa fe trail
                trail.load("santa fe")

            if event.key == pygame.K_q:
                pygame.quit()
                quit()


        if event.type == pygame.MOUSEBUTTONDOWN:
            Controller.pellet_on_click()

        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    
        game.update_state(screen, ant, map_, trail)
        game.update_score(ant, trail)
        # map_.guard(ant)
        # map_.draw(screen)
        # trail.draw(screen)
        # ant.draw(screen)
        # trail.update()

    pygame.display.update()

