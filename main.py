import pygame
import random
import os
import neat
import pickle
import sys
import matplotlib.pyplot as plt

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Define constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20
BALL_SIZE = 20
BLOCK_WIDTH = SCREEN_WIDTH // 10  # Ensuring full screen coverage
BLOCK_HEIGHT = 30
BLOCK_ROWS = 5
BLOCK_COLS = 10
BLOCK_COLORS = [RED, GREEN, BLUE]


class BlockBreakerGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Block Breaker")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)

        self.paddle = pygame.Rect((SCREEN_WIDTH - PADDLE_WIDTH) // 2, SCREEN_HEIGHT - PADDLE_HEIGHT - 10, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.ball = pygame.Rect(SCREEN_WIDTH // 2 - BALL_SIZE // 2, SCREEN_HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
        self.ball_velocity = [random.choice([-5, 5]), 5]  # Random initial velocity

        self.blocks = []
        self.create_blocks()
        self.score = 0

    def create_blocks(self):
        self.blocks.clear()
        for row in range(BLOCK_ROWS):
            for col in range(BLOCK_COLS):
                block_color = random.choice(BLOCK_COLORS)
                block = pygame.Rect(col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT)
                self.blocks.append((block, block_color))

    def update(self, net):
        # Get inputs for the neural network
        inputs = (self.paddle.x, self.ball.x, self.ball.y, self.ball_velocity[0], self.ball_velocity[1])
        output = net.activate(inputs)
        decision = output.index(max(output))

        # Move paddle based on the neural network's decision
        if decision == 0:  # Move left
            self.paddle.x -= 15
        elif decision == 1:  # Move right
            self.paddle.x += 15
        elif decision == 2:  # Do nothing
            pass

        # Ensure paddle stays within screen bounds
        if self.paddle.left < 0:
            self.paddle.left = 0
        if self.paddle.right > SCREEN_WIDTH:
            self.paddle.right = SCREEN_WIDTH

        # Update ball position
        self.ball.x += self.ball_velocity[0]
        self.ball.y += self.ball_velocity[1]

        # Ball collisions with walls
        if self.ball.left < 0 or self.ball.right > SCREEN_WIDTH:
            self.ball_velocity[0] *= -1
        if self.ball.top < 0:
            self.ball_velocity[1] *= -1

        # Ball collision with paddle
        if self.ball.colliderect(self.paddle):
            self.ball_velocity[1] *= -1

        # Ball collision with blocks
        for block, _ in self.blocks:
            if self.ball.colliderect(block):
                self.blocks.remove((block, _))
                self.ball_velocity[1] *= -1
                self.score += 1

        # Check for game over or win
        if self.ball.bottom >= SCREEN_HEIGHT:
            return True  # Game over
        if not self.blocks:
            print("You win!")
            return True  # Win condition
        return False

    def draw(self):
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, WHITE, self.paddle)
        pygame.draw.ellipse(self.screen, WHITE, self.ball)
        for block, color in self.blocks:
            pygame.draw.rect(self.screen, color, block)
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = BlockBreakerGame()

        while True:
            game.handle_events()
            game_over = game.update(net)
            game.draw()
            pygame.display.flip()
            game.clock.tick(240)  # Increase the game speed

            if game_over:
                break

        genome.fitness = game.score


def run_neat(config_path, load_winner=False):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    if load_winner and os.path.exists(selected_model + '.pkl'):
        with open(selected_model + '.pkl', 'rb') as f:
            winner = pickle.load(f)

    else:
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        best_fitnesses = []

        # Run indefinitely until manually stopped
        generation = 0
        while True:
            generation += 1
            winner = p.run(eval_genomes, 1)

            # Save the winner after each generation with unique filename
            with open(f'winner_gen_{generation}.pkl', 'wb') as f:
                pickle.dump(winner, f)
            best_fitnesses.append(winner.fitness)

            print(f'\nBest genome after generation {generation}:\n{winner}')
            plt.plot(best_fitnesses)
            plt.xlabel('Generation')
            plt.ylabel('Best Fitness')
            plt.title('Best Fitness Over Generations')
            plt.savefig('fitness_plot.png')
            plt.show()  # Add this line to display the plot window
            plt.close()
    # Play the game with the best network
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    game = BlockBreakerGame()

    while True:
        game.handle_events()
        game_over = game.update(winner_net)
        game.draw()
        pygame.display.flip()
        game.clock.tick(60)

        if game_over:
            break


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'ffs.txt')

    choice = input("Do you want to load the saved model? (y/n): ")
    if choice.lower() == 'y':
        selected_model = input("Please enter the model name: ")  # Prompt for the model name
        run_neat(config_path, load_winner=True)
    else:
        run_neat(config_path, load_winner=False)
