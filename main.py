import pygame
import neat
import os
import random
import pickle

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FOOD_COUNT = 20
OBSTACLE_COUNT = 10

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.food = []
        self.obstacles = []
        self.death_zones = []

    def generate_food(self):
        self.food = []
        for _ in range(FOOD_COUNT):
            x = random.randint(0, self.width - 10)
            y = random.randint(0, self.height - 10)
            self.food.append(pygame.Rect(x, y, 10, 10))

    def generate_obstacles(self):
        self.obstacles = []
        for _ in range(OBSTACLE_COUNT):
            x = random.randint(0, self.width - 50)
            y = random.randint(0, self.height - 50)
            self.obstacles.append(pygame.Rect(x, y, 50, 50))

    def generate_death_zones(self):
        self.death_zones = []
        for _ in range(OBSTACLE_COUNT):
            x = random.randint(0, self.width - 100)
            y = random.randint(0, self.height - 100)
            self.death_zones.append(pygame.Rect(x, y, 100, 100))  # Adjust size to 100x100

    def draw(self, screen):
        for food in self.food:
            pygame.draw.rect(screen, (0, 255, 0), food)
        for obstacle in self.obstacles:
            pygame.draw.rect(screen, (255, 0, 0), obstacle)
        for death_zone in self.death_zones:
            pygame.draw.rect(screen, (0, 0, 0), death_zone)

class Agent:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.energy = 100
        self.alive = True

    def move(self, action):
        if action == 0:  # Up
            self.rect.y -= 5
        elif action == 1:  # Down
            self.rect.y += 5
        elif action == 2:  # Left
            self.rect.x -= 5
        elif action == 3:  # Right
            self.rect.x += 5

        # Keep the agent within the screen bounds
        if self.rect.x < 0:
            self.rect.x = 0
        elif self.rect.x > SCREEN_WIDTH - self.rect.width:
            self.rect.x = SCREEN_WIDTH - self.rect.width

        if self.rect.y < 0:
            self.rect.y = 0
        elif self.rect.y > SCREEN_HEIGHT - self.rect.height:
            self.rect.y = SCREEN_HEIGHT - self.rect.height

    def update(self, environment):
        if not self.alive:
            return

        food_index = self.rect.collidelist(environment.food)
        if food_index != -1:
            environment.food.pop(food_index)  # Remove the collided food
            self.energy += 50  # Increase energy on eating food

        obstacle_index = self.rect.collidelist(environment.obstacles)
        if obstacle_index != -1:
            self.energy -= 10  # Decrease energy on hitting an obstacle

        death_zone_index = self.rect.collidelist(environment.death_zones)
        if death_zone_index != -1:
            self.alive = False  # Kill agent in death zone

        self.energy -= 1  # Energy decreases over time
        if self.energy <= 0:
            self.alive = False  # Agent dies if energy depletes

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 0, 255), self.rect)

def eval_genomes(genomes, config):
    width, height = SCREEN_WIDTH, SCREEN_HEIGHT
    env = Environment(width, height)
    env.generate_food()
    env.generate_obstacles()
    env.generate_death_zones()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    agents = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        agent = Agent(width // 2, height // 2)
        agents.append((agent, genome, net))

    run_simulation = True
    while run_simulation:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill((255, 255, 255))
        env.draw(screen)

        all_dead = True  # Reset flag at the start of each generation
        for agent, genome, net in agents:
            if agent.alive:
                all_dead = False  # Set to False if any agent is alive
                inputs = [agent.rect.x, agent.rect.y, agent.energy]
                for food in env.food:
                    inputs.extend([food.x, food.y])
                for obstacle in env.obstacles:
                    inputs.extend([obstacle.x, obstacle.y])
                for death_zone in env.death_zones:
                    inputs.extend([death_zone.x, death_zone.y])

                # Ensure the number of inputs matches the expected size by padding with zeros if necessary
                while len(inputs) < 83:
                    inputs.append(0)

                action = net.activate(inputs)
                agent.move(action.index(max(action)))
                agent.update(env)
                genome.fitness += agent.energy

            agent.draw(screen)

        pygame.display.update()
        clock.tick(30)

        if all_dead:
            run_simulation = False  # End simulation if all agents are dead

def run_neat(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_file)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 50)

    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)

    print(f'\nBest genome:\n{winner}')

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run_neat(config_path)
