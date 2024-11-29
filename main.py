import pygame
import sys
import time
import csv
from GeneticAlgorithm import GeneticAlgorithm
from Track import Track

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60
GENERATION_TIME_LIMIT = 15  # Time limit in seconds for each generation

# Experiment Parameters
population_sizes = [20, 50, 100]  # Different population sizes to test
mutation_rates = [0.01, 0.05, 0.1]  # Different mutation rates to test
num_generations = 30  # Number of generations to run for each experiment

# Results file
results_file = "experiment_results.csv"


def run_experiment(pop_size, mut_rate, track, writer):
    """Runs a single experiment with the given population size and mutation rate."""
    ga = GeneticAlgorithm(population_size=pop_size)

    for gen in range(num_generations):
        for _ in range(FPS * GENERATION_TIME_LIMIT):  # Simulate for the generation time limit
            for car in ga.cars:
                if car.alive:
                    car.update(track)  # Simulate the car's behavior on the track

        # Perform genetic algorithm steps
        ga.start_new_generation()

        # Record the best fitness of this generation
        writer.writerow([pop_size, mut_rate, gen + 1, ga.best_fitness])

        # Display progress in the terminal
        print(f"Pop: {pop_size}, Mut: {mut_rate}, Gen: {gen + 1}, Best Fitness: {ga.best_fitness:.2f}")


def main():
    # Set up screen and track
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Self-Driving Car Evolution")
    clock = pygame.time.Clock()
    track = Track()

    # Open CSV file to store results
    with open(results_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Population Size", "Mutation Rate", "Generation", "Best Fitness"])

        # Run experiments for all combinations of population sizes and mutation rates
        for pop_size in population_sizes:
            for mut_rate in mutation_rates:
                print(f"Running experiment with Pop: {pop_size}, Mut: {mut_rate}")
                run_experiment(pop_size, mut_rate, track, writer)

    print(f"Experiments completed! Results saved to {results_file}")

    # Visualization of the last experiment
    running = True
    ga = GeneticAlgorithm(population_size=population_sizes[-1])

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
        screen.fill((0, 0, 0))  # Black background
        track.draw(screen)
        for car in ga.cars:
            car.draw(screen)

        # Draw generation counter and timer
        font = pygame.font.Font(None, 36)
        gen_text = font.render(f"Generation: {ga.generation}", True, (255, 255, 255))
        time_text = font.render(f"Time: {time_remaining:.1f}s", True, (255, 255, 0))
        fitness_text = font.render(f"Best Fitness: {ga.best_fitness:.0f}", True, (0, 255, 0))
        screen.blit(gen_text, (10, 10))
        screen.blit(time_text, (10, 50))
        screen.blit(fitness_text, (10, 90))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
