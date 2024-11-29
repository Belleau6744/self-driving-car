import pygame
import sys
import time

from GeneticAlgorithm import GeneticAlgorithm
from Track import Track

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60
GENERATION_TIME_LIMIT = 15  # Time limit in seconds for each generation

# Updated starting position for cars
START_X = 600  # Center of track horizontally
START_Y = 600  # Slightly above bottom middle of track
START_ANGLE = 180  # Facing upward (270 degrees points up in pygame coordinates)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (147, 112, 219)
GRAY = (128, 128, 128)


def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Self-Driving Car Evolution")
    clock = pygame.time.Clock()

    track = Track()
    ga = GeneticAlgorithm()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        current_time = time.time()
        generation_time = current_time - ga.generation_start_time
        time_remaining = max(0, GENERATION_TIME_LIMIT - generation_time)

        # Check if generation time limit is reached or all cars are dead
        all_dead = all(not car.alive for car in ga.cars)
        if time_remaining <= 0 or all_dead:
            ga.start_new_generation()

        # Update
        for car in ga.cars:
            if car.alive:
                car.update(track)

        # Draw
        screen.fill(BLACK)
        track.draw(screen)
        for car in ga.cars:
            car.draw(screen)

        # Draw generation counter and timer
        font = pygame.font.Font(None, 36)
        gen_text = font.render(f"Generation: {ga.generation}", True, WHITE)
        time_text = font.render(f"Time: {time_remaining:.1f}s", True, YELLOW)
        fitness_text = font.render(f"Best Fitness: {ga.best_fitness:.0f}", True, GREEN)
        screen.blit(gen_text, (10, 10))
        screen.blit(time_text, (10, 50))
        screen.blit(fitness_text, (10, 90))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
