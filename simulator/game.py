# Simulation environment for Neural networks performing foraging tasks
# Author(s) : Daniel Monahan
# contact(s) : danielkm@github.com
#
# 


# WIP Currently porting controls, object classes, and trail functionality from trail-game
import pygame
import time

GRID_SIZE = 32

# window size
SCALE = 20
WIDTH = GRID_SIZE * SCALE
HEIGHT = GRID_SIZE * SCALE

TOROIDAL_MAP = True     # if true edges, are wrapped around to eachother. otherwise edges are walls

SANTA_FE = [[0.0, 0.0], [20.0, 0.0], [40.0, 0.0], [60.0, 0.0], [60.0, 20.0], [60.0, 40.0], [60.0, 60.0], [60.0, 80.0], [60.0, 100.0], [80.0, 100.0], [100.0, 100.0], [120.0, 100.0], [160.0, 100.0], [180.0, 100.0], [200.0, 100.0], [220.0, 100.0], [240.0, 100.0], [240.0, 120.0], [240.0, 140.0], [240.0, 160.0], [240.0, 180.0], [240.0, 200.0], [240.0, 240.0], [240.0, 260.0], [240.0, 280.0], [240.0, 300.0], [240.0, 360.0], [240.0, 380.0], [240.0, 400.0], [240.0, 420.0], [240.0, 440.0], [240.0, 460.0], [220.0, 480.0], [200.0, 480.0], [180.0, 480.0], [160.0, 480.0], [140.0, 480.0], [100.0, 480.0], [80.0, 480.0], [40.0, 500.0], [40.0, 520.0], [40.0, 540.0], [40.0, 560.0], [60.0, 600.0], [80.0, 600.0], [100.0, 600.0], [120.0, 600.0], [160.0, 580.0], [160.0, 560.0], [180.0, 540.0], [200.0, 540.0], [220.0, 540.0], [240.0, 540.0], [260.0, 540.0], [280.0, 540.0], [300.0, 540.0], [340.0, 520.0], [340.0, 500.0], [340.0, 480.0], [340.0, 420.0], [340.0, 380.0], [340.0, 360.0], [340.0, 340.0], [360.0, 320.0], [420.0, 300.0], [420.0, 280.0], [420.0, 220.0], [420.0, 200.0], [420.0, 180.0], [420.0, 160.0], [440.0, 100.0], [460.0, 100.0], [500.0, 80.0], [500.0, 60.0], [520.0, 40.0], [540.0, 40.0], [560.0, 40.0], [580.0, 60.0], [580.0, 80.0], [580.0, 120.0], [580.0, 180.0], [580.0, 240.0], [560.0, 280.0], [540.0, 280.0], [520.0, 280.0], [460.0, 300.0], [480.0, 360.0], [540.0, 380.0], [520.0, 440.0], [460.0, 460.0]]

# colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
gray = pygame.Color(150, 150, 150)
red = pygame.Color(125, 10, 10)
lightyellow = pygame.Color(250, 245, 200)

agent = pygame.image.load('assets/agent.png')
agent = pygame.transform.scale(agent, (WIDTH//GRID_SIZE, HEIGHT//GRID_SIZE))

class Ant:
    def __init__(self, position=[0, 0]):
        self.position = position
        self.dir = 0 # 0 = right, 1 = up, 2 = left, 3 = down

    def move(self):
        if (self.dir == 0):
            self.position[0] += WIDTH // GRID_SIZE
        elif (self.dir == 1):
            self.position[1] -= HEIGHT // GRID_SIZE
        elif (self.dir == 2):
            self.position[0] -= WIDTH // GRID_SIZE
        elif (self.dir == 3):
            self.position[1] += HEIGHT // GRID_SIZE
        else :
            raise ValueError
            print("Ant.dir is out of range!")

    def draw(self):
        sprite = pygame.transform.rotate(agent, self.dir * (360/4))
        map_.surface.blit(sprite, (self.position[0], self.position[1]) )


class Map:
    def __init__(self, width=WIDTH, height=HEIGHT, color=lightyellow, toroidal=TOROIDAL_MAP):
        self.width = width
        self.height = height
        self.color = color
        self.toroidal = toroidal

        self.surface = pygame.display.set_mode((self.width, self.height))
    
    def handleEdges(self):
        if ant.position[0] >= self.width:
            ant.position[0] = 0 if self.toroidal else self.width - self.width/GRID_SIZE

        if ant.position[0] < 0:
            ant.position[0] = self.width - self.width/GRID_SIZE if self.toroidal else 0

        if ant.position[1] >= self.height:
            ant.position[1] = 0 if self.toroidal else self.height - self.height/GRID_SIZE
        
        if ant.position[1] < 0:
            ant.position[1] = self.height - self.height/GRID_SIZE if self.toroidal else 0

    def draw(self):
        self.surface.fill(self.color)
        
        for x in range(0, self.width, self.width // GRID_SIZE):
            for y in range(0, self.height, self.height // GRID_SIZE):
                pygame.draw.line(self.surface, gray, (x, 0), (x, self.height), 2)
                pygame.draw.line(self.surface, gray, (0, y), (self.width, y), 2)


class Pellet:
    def __init__(self, position=[WIDTH/GRID_SIZE * 10, HEIGHT/GRID_SIZE * 10]):
        self.position = position
        self.wasEaten = False

    def draw(self):
        pygame.draw.rect(map_.surface, red, pygame.Rect(self.position[0], self.position[1], 
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
               self.pellets.pop(i-1)

    def draw(self):
        for i in range(len(self.pellets)):
            self.pellets[i].draw()


# Initialising pygame
pygame.init()

# FPS (frames per second) controller
fps = 120


def rlm_controls(ant, key):
    if (key == pygame.K_j):
        ant.dir = (ant.dir + 1) % 4
    elif (key == pygame.K_k):
        ant.dir = (ant.dir - 1) % 4
    elif (key == pygame.K_f):
        ant.move()
    return

def pellet_on_click():
    mousePosition = pygame.mouse.get_pos()
    xPos = mousePosition[0] - mousePosition[0] % (WIDTH / GRID_SIZE);
    yPos = mousePosition[1] - mousePosition[1] % (HEIGHT / GRID_SIZE);
    trail.addPellet([xPos, yPos]);


font = pygame.font.Font
ant = Ant();
map_ = Map();
trail = Trail();

while True:

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            rlm_controls(ant, event.key)

            if event.key == pygame.K_s: # load santa fe trail
                trail.load("santa fe")

            if event.key == pygame.K_q:
                pygame.quit()
                quit()


        if event.type == pygame.MOUSEBUTTONDOWN:
            pellet_on_click()

        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    
    map_.handleEdges()
    map_.draw()
    trail.draw()
    ant.draw()

    pygame.display.update()

    trail.update()

