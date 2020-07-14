import pygame
from random import randint
import neat
import os
import pickle
import datetime
import gzip
import visualize

WIDTH = 480
HEIGHT = 360
gen = 0
winON = True

pygame.init()
STAT_FONT = pygame.font.SysFont("comicsans", 40)
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Genetic Pong")

class Paddle:

    def __init__(self, x, y, color):
        """
        Initialize the object
        :param x: starting x pos (int)
        :param y: starting y pos (int)
        :return: None
        """
        self.x = x
        self.y = y
        self.vel = 0
        self.color = color
        self.width = 5
        self.height = HEIGHT/10
        self.rect = pygame.Rect(self.x,self.y,self.width,self.height)

    def moveUP(self):
        """
        make the object go up
        :return: None
        """
        self.vel = -5
    
    def moveDOWN(self):
        """
        make the object go down
        :return: None
        """
        self.vel = 5
    
    def moveSTOP(self):
        """
        makes the object stop moving
        :return: None
        """
        self.vel = 0

    def move(self):
        if self.rect.top <= 0 and self.vel < 0:
            self.vel = 0
        elif self.rect.bottom >= HEIGHT and self.vel > 0:
            self.vel = 0

        self.rect = self.rect.move([0,self.vel])

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
    
    def getY(self):
        return self.rect.y
    
    def getX(self):
        return self.rect.x

class Ball:

    def __init__(self, x, y, color):
        """
        Initialize the object
        :param x: starting x pos (int)
        :param y: starting y pos (int)
        :return: None
        """
        self.x = x
        self.y = y
        self.vel = [randomSign()*4,randomSign()*4]
        self.color = color
        self.width = 10
        self.rect = pygame.Rect(self.x,self.y,self.width,self.width)

    def changeVelY(self):
        """
        changes the objects y direction
        :return: None
        """
        self.vel[1] = -self.vel[1]
    
    def changeVelX(self):
        """
        changes the objects x direction
        :return: None
        """
        self.vel[0] = -self.vel[0]

    def move(self):
        #if ball hits the bottom or top of screen change y direction
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.changeVelY()
        self.rect = self.rect.move([self.vel[0],self.vel[1]])

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

    def getX(self):
        return self.rect.x

    def getY(self):
        return self.rect.y
    
    def collide(self, paddle):
        return self.rect.colliderect(paddle)

def drawWindow(screen, lPaddles, rPaddle, balls):
    """
    draws the windows for the main game loop
    :param screen: pygame window surface
    :param lPaddles: a list of Left Paddles
    :param rPaddle: the Right Paddle
    :param balls: a list of balls
    :return: None
    """
    screen.fill((0,0,0))

    for ball in balls:
        ball.draw(screen)

    for paddle in lPaddles:
        paddle.draw(screen)

    rPaddle.draw(screen)
    pygame.display.flip()

def drawWindow(screen, paddles, paddlesR, balls):
    """
    draws the windows for the main game loop
    :param screen: pygame window surface
    :param paddles: a list of Left Paddles
    :param balls: a list of balls
    :return: None
    """
    screen.fill((0,0,0))

    for ball in balls:
        ball.draw(screen)

    for paddle in paddles:
        paddle.draw(screen)

    for paddle in paddlesR:
        paddle.draw(screen)

    # generations
    score_label = STAT_FONT.render("Gens: " + str(gen-1),1,(255,255,255))
    screen.blit(score_label, (10, 10))

    pygame.display.flip()

def save_object(obj, filename):
    with gzip.open(filename, 'w', compresslevel=5) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with gzip.open(filename) as f:
        obj = pickle.load(f)
        return obj

def randomSign():
    i = randint(0,1)
    if i == 0: 
        return -1
    else: return 1

def main(screen, lPaddles, rPaddle, balls):

    clock = pygame.time.Clock()

    run = True
    while run and len(lPaddles) > 0:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w: lPaddles[0].moveUP()
                if event.key == pygame.K_s: lPaddles[0].moveDOWN()
            else:
                lPaddles[0].moveSTOP()

            if event.type == pygame.KEYDOWN:
                #right move
                if event.key == pygame.K_UP: rPaddle.moveUP()
                if event.key == pygame.K_DOWN: rPaddle.moveDOWN()        
            else:
                rPaddle.moveSTOP()
        

        rPaddle.move()

        
        for ball in balls:
            if ball.collide(lPaddles[balls.index(ball)]):
                ball.changeVelX()
                lPaddles[balls.index(ball)].fitness += 5
            ball.move()
            if ball.getX() < 0:
                #print(balls.index(ball))
                lPaddles[balls.index(ball)].fitness = 1
                lPaddles.pop(balls.index(ball))
                balls.pop(balls.index(ball))
        
        for paddle in lPaddles:
            paddle.move()
            paddle.fitness += 1

        drawWindow(screen, lPaddles, rPaddle, balls)

def eval_genomes(genomes, config):
    global gen
    global winON
    gen += 1
    score = 0

    paddles = []
    paddlesR = []
    balls = []
    nets = [] #holds all the differnet NN
    ge = [] # keeps track of genomes

    #for every genome set up a ball, paddle, and NN for it and put in ge to keep track
    for genome_id, g in genomes:
        tmp_color = (randint(100,255),randint(100,255),randint(100,255))
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        paddles.append(Paddle(0,240,tmp_color))
        paddlesR.append(Paddle(WIDTH-5,240,tmp_color))
        balls.append(Ball(randint(100,255),randint(100,255),tmp_color))
        g.fitness = 0
        ge.append(g)
    
    clock = pygame.time.Clock()

    run = True
    while run and len(paddles) > 0:
        if winON: clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break
        #left paddle
        for x, paddle in enumerate(paddles):
            ge[x].fitness += 0.05
            paddle.move()

            #send the inputs to the NNs and receive output and decide if move up, down, or not move
            #TODO: test out differnt inputs to the ANN, try abs(paddle.y-ball.y) instead of ball.y or other variantions
            outputs = nets[paddles.index(paddle)].activate((paddle.getY(),
                                                            abs(paddle.getX() - balls[paddles.index(paddle)].rect.x),
                                                            balls[paddles.index(paddle)].rect.y))

            if outputs[0] > outputs[1]:
                if outputs[0] > 0.5:
                    paddle.moveUP()
                else:
                    paddle.moveSTOP()
            elif outputs[1] > 0.5:
                paddle.moveDOWN()
            else:
                paddle.moveSTOP()
        #right paddle
        for x, paddle in enumerate(paddlesR):
            ge[x].fitness += 0.05
            paddle.move()

            outputs = nets[paddlesR.index(paddle)].activate((paddle.getY(),
                                                            abs(paddle.getX() - balls[paddlesR.index(paddle)].rect.x),
                                                            balls[paddlesR.index(paddle)].rect.y))
            
            if outputs[0] > outputs[1]:
                if outputs[0] > 0.5:
                    paddle.moveUP()
                else:
                    paddle.moveSTOP()
            elif outputs[1] > 0.5:
                paddle.moveDOWN()
            else:
                paddle.moveSTOP()
        
        
        for ball in balls:
            if ball.collide(paddles[balls.index(ball)]):
                ball.changeVelX()
                ge[balls.index(ball)].fitness += 5
                score += 1
            if ball.collide(paddlesR[balls.index(ball)]):
                ball.changeVelX()
                ge[balls.index(ball)].fitness += 5
                score += 1
            ball.move()
            if ball.getX() < 0 or ball.getX() > WIDTH:
                #check if paddle misses ball
                ge[balls.index(ball)].fitness -= 2
                nets.pop(balls.index(ball))
                ge.pop(balls.index(ball))
                paddles.pop(balls.index(ball))
                paddlesR.pop(balls.index(ball))
                balls.pop(balls.index(ball))
        #render / update screen
        if winON: drawWindow(screen, paddles, paddlesR, balls)

        if score > 500:
            break

def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play pong.
    :param config_file: location of config file
    :return: None
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
    checkpoint = neat.Checkpointer(5, "test")
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 1000)
    #checkpoint.save_checkpoint()

    #save_object(winner, f"winner{str(datetime.datetime.now().time()).split('.')[0]+'_'+str(datetime.datetime.now().date())}.pkl")

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    #visualize.draw_net(config, winner, True, )
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)

    #lpad = Paddle(0,200)
    #l2pad = Paddle(0,100)
    #rpad = Paddle(WIDTH-5,200)
    #ball1 = Ball(100, 300)
    #ball2 = Ball(240, 200)
    #game1 = Game()
    #main(screen,[lpad,l2pad],rpad,[ball1,ball2])

