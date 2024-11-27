import pygame
from GeneticAlgorithm import GeneticAlgorithm

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WAYPOINTS = [
    (150, 500),
    (650, 500),
    (650, 300),
    (150, 300),
    (150, 100),
    (650, 100),
]  # Waypoints defining the track
FPS = 60
GENERATION_TIME = 3  # Seconds per generation

# Screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Self-Driving Car Simulation with Track")

# Track Surface
track = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
track.fill((255, 255, 255))  # White background
pygame.draw.rect(track, (0, 0, 0), (50, 50, 700, 500), width=10)  # Outer border


def main():
    clock = pygame.time.Clock()
    ga = GeneticAlgorithm()
    running = True
    start_time = pygame.time.get_ticks()
    best_overall_fitness = float('-inf')  # Track the best fitness across generations

    while running:
        screen.blit(track, (0, 0))  # Draw the track background

        # Draw waypoints
        for i, (x, y) in enumerate(WAYPOINTS):
            pygame.draw.circle(screen, (0, 255, 0), (x, y), 10)
            if i > 0:
                pygame.draw.line(screen, (0, 255, 255), WAYPOINTS[i - 1], (x, y), 2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update and draw cars
        for car in ga.population:
            car.update(track)
            car.draw()

        # Calculate stats
        alive_cars = sum(car.alive for car in ga.population)
        best_generation_fitness = max(car.fitness for car in ga.population)
        average_generation_fitness = sum(car.fitness for car in ga.population) / len(ga.population)
        best_overall_fitness = max(best_overall_fitness, best_generation_fitness)

        # Create a transparent stats panel
        font = pygame.font.SysFont(None, 24)
        stats_surface = pygame.Surface((200, SCREEN_HEIGHT), pygame.SRCALPHA)  # Enable transparency
        stats_surface.fill((200, 200, 200, 100))  # Light gray with low opacity (100 for alpha)

        # Render stats text
        y_offset = 10

        def render_stat(label, value):
            """Render a single stat line."""
            nonlocal y_offset
            stat_text = font.render(f"{label}: {value}", True, (0, 0, 0))  # Black text
            stats_surface.blit(stat_text, (10, y_offset))
            y_offset += 30  # Space between lines

        render_stat("Generation", ga.generation)
        render_stat("Alive Cars", alive_cars)
        render_stat("Best Gen Fitness", round(best_generation_fitness, 2))
        render_stat("Best Overall Fitness", round(best_overall_fitness, 2))
        render_stat("Avg Gen Fitness", round(average_generation_fitness, 2))

        # Position the stats panel on the right of the screen
        screen.blit(stats_surface, (SCREEN_WIDTH - 200, 0))

        # Check if all cars are dead or generation time is up
        if all(not car.alive for car in ga.population) or (
            pygame.time.get_ticks() - start_time > GENERATION_TIME * 1000
        ):
            ga.evolve()
            start_time = pygame.time.get_ticks()

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
