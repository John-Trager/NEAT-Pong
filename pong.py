"""
Ai Pong Trainer using NEAT learning algorithm.

@author: John Trager

2020
"""
import gzip
import sys
import os
import pickle
from random import randint
from pygame import QUIT, init
import pygame
import neat


WIDTH = 480
HEIGHT = 360
GEN = 0
WIN_ON = True

init()
STAT_FONT = pygame.font.SysFont("comicsans", 40)
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Genetic Pong")

class Paddle:
    """
    Paddle Object
    """

    def __init__(self, x_c, y_c, color):
        """
        Initialize the object
        :param x_c: starting x_c pos (int)
        :param y_c: starting y_c pos (int)
        :return: Void
        """
        self.x_c = x_c
        self.y_c = y_c
        self.vel = 0
        self.color = color
        self.width = 5
        self.height = HEIGHT/10
        self.rect = pygame.Rect(self.x_c,self.y_c,self.width,self.height)

    def move_up(self):
        """
        make the object go up
        :return: Void
        """
        self.vel = -5

    def move_down(self):
        """
        make the object go down
        :return: Void
        """
        self.vel = 5

    def move_stop(self):
        """
        makes the object stop moving
        :return: Void
        """
        self.vel = 0

    def move(self):
        """
        moves paddle if not colliding with top or bottom of screen
        """
        if self.rect.top <= 0 and self.vel < 0:
            self.vel = 0
        elif self.rect.bottom >= HEIGHT and self.vel > 0:
            self.vel = 0

        self.rect = self.rect.move([0,self.vel])

    def draw(self, screen):
        """
        draws the pygram rects ontp the screen
        """
        pygame.draw.rect(screen, self.color, self.rect)

    def get_y(self):
        """
        returns the Y coordinate of the paddle
        """
        return self.rect.y

    def get_x(self):
        """
        returns the X coordinate of the paddle
        """
        return self.rect.x

class Ball:
    """
    Ball Object
    """

    def __init__(self, x_c, y_c, color):
        """
        Initialize the object
        :param x_c: starting x_c pos (int)
        :param y_c: starting y_c pos (int)
        :return: Void
        """
        self.x_c = x_c
        self.y_c = y_c
        self.vel = [random_sign()*4,random_sign()*4]
        self.color = color
        self.width = 10
        self.rect = pygame.Rect(self.x_c,self.y_c,self.width,self.width)

    def change_vel_y(self):
        """
        changes the objects y_c direction
        :return: Void
        """
        self.vel[1] = -self.vel[1]

    def change_vel_x(self):
        """
        changes the objects x_c direction
        :return: Void
        """
        self.vel[0] = -self.vel[0]

    def move(self):
        """
        moves ball object (with collision)
        """
        #if ball hits the bottom or top of screen change y_c direction
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.change_vel_y()
        self.rect = self.rect.move([self.vel[0],self.vel[1]])

    def draw(self, screen):
        """
        Draws the rect object onto screen
        """
        pygame.draw.rect(screen, self.color, self.rect)

    def get_x(self):
        """
        returns Ball object X coordinate
        """
        return self.rect.x

    def get_y(self):
        """
        returns Ball object Y coordinate
        """
        return self.rect.y

    def collide(self, paddle):
        """
        Returns true if any portion of either rectangle overlap
        (except the top+bottom or left+right edges).
        """
        return self.rect.colliderect(paddle)

def draw_window(screen, paddles, paddles_r, balls):
    """
    draws the windows for the main game loop
    :param screen: pygame window surface
    :param paddles: a list of Left Paddles
    :param balls: a list of balls
    :return: None
    """
    #draws black screen
    screen.fill((0,0,0))

    #draws balls
    for ball in balls:
        ball.draw(screen)

    #draws left side paddles
    for paddle in paddles:
        paddle.draw(screen)

    #draws right side paddles
    for paddle in paddles_r:
        paddle.draw(screen)

    # generations
    score_label = STAT_FONT.render("Gens: " + str(GEN-1),1,(255,255,255))
    screen.blit(score_label, (10, 10))

    pygame.display.flip()

def save_object(obj, filename):
    """
    saves model into .pkl file
    """
    with gzip.open(filename, 'w', compresslevel=5) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    """
    reads model from .pkl file and returns object
    TODO: as of 7-14-20 this function has had some problems that need to be sorted
    """
    with gzip.open(filename) as f:
        obj = pickle.load(f)
        return obj

def random_sign():
    """
    returns a random sign either -1 or +1
    """
    i = randint(0,1)
    if i == 0:
        return -1
    return 1

def eval_genomes(genomes, config):
    """
    evolves the different genomes using NEAT algorithm
    handles collisions and movement of paddles and balls
    """
    global GEN
    global WIN_ON
    GEN += 1
    score = 0

    paddles = []
    paddles_r = []
    balls = []
    nets = [] #holds all the differnet NN
    ge = [] # keeps track of genomes

    #for every genome set up a ball, paddle, and NN for it and put in ge to keep track
    for genome_id, g in genomes:
        tmp_color = (randint(100,255),randint(100,255),randint(100,255))
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        paddles.append(Paddle(0,240,tmp_color))
        paddles_r.append(Paddle(WIDTH-5,240,tmp_color))
        balls.append(Ball(randint(100,255),randint(100,255),tmp_color))
        g.fitness = 0
        ge.append(g)

    clock = pygame.time.Clock()

    run = True
    while run and len(paddles) > 0:
        if WIN_ON: clock.tick(30)
        for event in pygame.event.get():
            if event.type == QUIT:
                run = False
                sys.exit()
                break
        #left paddle
        for x_c, paddle in enumerate(paddles):
            #given fitness for being alive
            ge[x_c].fitness += 0.05
            paddle.move()

            # send the inputs to the NNs and receive output and
            # decide if move up, down, or not move
            # todo: test out different inputs to the ANN,
            # try abs(paddle.y_c-ball.y_c) instead of ball.y_c or other variantions
            outputs = nets[paddles.index(paddle)].activate((paddle.get_y(),
                                    abs(paddle.get_x() - balls[paddles.index(paddle)].rect.x),
                                    balls[paddles.index(paddle)].rect.y))

            if outputs[0] > outputs[1]:
                if outputs[0] > 0.5:
                    paddle.move_up()
                else:
                    paddle.move_stop()
            elif outputs[1] > 0.5:
                paddle.move_down()
            else:
                paddle.move_stop()
        #right paddle
        for x_c, paddle in enumerate(paddles_r):
            #given fitness for being alive
            ge[x_c].fitness += 0.05
            paddle.move()

            outputs = nets[paddles_r.index(paddle)].activate((paddle.get_y(),
                            abs(paddle.get_x() - balls[paddles_r.index(paddle)].rect.x),
                            balls[paddles_r.index(paddle)].rect.y))

            if outputs[0] > outputs[1]:
                if outputs[0] > 0.5:
                    paddle.move_up()
                else:
                    paddle.move_stop()
            elif outputs[1] > 0.5:
                paddle.move_down()
            else:
                paddle.move_stop()

        #gives fitness to evolution if its corresponding ball
        for ball in balls:
            if ball.collide(paddles[balls.index(ball)]):
                ball.change_vel_x()
                ge[balls.index(ball)].fitness += 5
                score += 1
            if ball.collide(paddles_r[balls.index(ball)]):
                ball.change_vel_x()
                ge[balls.index(ball)].fitness += 5
                score += 1
            ball.move()
            if ball.get_x() < 0 or ball.get_x() > WIDTH:
                #check if paddle misses ball
                ge[balls.index(ball)].fitness -= 2
                nets.pop(balls.index(ball))
                ge.pop(balls.index(ball))
                paddles.pop(balls.index(ball))
                paddles_r.pop(balls.index(ball))
                balls.pop(balls.index(ball))

        #render / update screen
        if WIN_ON:
            draw_window(screen, paddles, paddles_r, balls)

        #breaks the evolution if the score reaches 500
        if score > 500:
            break

def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play pong.
    :param config_file: location of config file
    :return: Void
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    #Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #checkpoint = nexat.Checkpointer(5, "test")
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 1000)
    #checkpoint.save_checkpoint()

    #save_object(winner, f"winner{str(datetime.datetime.now().time()).split('.')[0]+'_'+
    # str(datetime.datetime.now().date())}.pkl")

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)

