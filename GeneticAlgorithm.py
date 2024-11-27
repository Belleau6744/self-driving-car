import pygame
import numpy as np
import random

from Car import Car

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
NUM_CARS = 30
MUTATION_RATE = 0.1
NUM_SENSORS = 5  # Original sensors, plus one for track proximity

class GeneticAlgorithm:
    """Class representing the genetic algorithm to evolve the cars."""
    def __init__(self):
        # Initialize the first generation with a population of cars
        self.population = [Car() for _ in range(NUM_CARS)]
        self.generation = 1

    def evolve(self):
        """Evolve the population to the next generation using selection, crossover, and mutation."""
        self.generation += 1
        fitness = [car.fitness for car in self.population]

        # Debugging: Print average and best fitness
        print(
            f"Generation {self.generation}: Best fitness = {max(fitness)}, Average fitness = {sum(fitness) / len(fitness)}"
        )

        # Normalize fitness to avoid negative values
        min_fitness = min(fitness)
        fitness = [f - min_fitness + 1 for f in fitness]  # Ensure all values are positive

        # Calculate selection probabilities based on normalized fitness
        total_fitness = sum(fitness)
        probabilities = [f / total_fitness for f in fitness]

        # Elitism: Keep the top 10% of the population
        elite_count = max(1, NUM_CARS // 10)
        elites = sorted(self.population, key=lambda car: car.fitness, reverse=True)[:elite_count]

        # Select parents for the next generation
        parents = random.choices(self.population, probabilities, k=NUM_CARS - elite_count)

        # Generate new offspring through crossover and mutation
        new_population = []
        for _ in range((NUM_CARS - elite_count) // 2):
            p1, p2 = random.sample(parents, 2)
            child1, child2 = self.crossover(p1, p2)
            new_population.append(self.mutate(child1))
            new_population.append(self.mutate(child2))

        # Add elites to the new population to preserve the best performers
        new_population.extend(elites)

        # Replace the old population with the new one
        self.population = new_population

        # Reset all cars for the next generation
        for car in self.population:
            car.reset()

    def crossover(self, p1, p2):
        """Create two offspring by combining the brains of two parent cars."""
        child1 = Car()
        child2 = Car()

        # Choose a random crossover point
        crossover_point = random.randint(0, NUM_SENSORS)  # Include the extra sensor

        # Combine parent brains at the crossover point
        child1.brain[:crossover_point] = p1.brain[:crossover_point]
        child1.brain[crossover_point:] = p2.brain[crossover_point:]
        child2.brain[:crossover_point] = p2.brain[:crossover_point]
        child2.brain[crossover_point:] = p1.brain[crossover_point:]
        return child1, child2

    def mutate(self, car):
        """Introduce random changes to a car's brain to encourage diversity."""
        if random.random() < MUTATION_RATE:
            mutation_matrix = np.random.uniform(-0.1, 0.1, car.brain.shape)
            car.brain += mutation_matrix
        return car
