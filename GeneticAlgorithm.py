import pygame
import numpy as np
from random import random, randint, choice
import time

from Car import Car

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


class GeneticAlgorithm:
    def __init__(self, population_size=50):
        self.population_size = population_size
        self.generation = 0
        self.cars = [Car() for _ in range(population_size)]
        self.generation_start_time = time.time()
        self.best_fitness = 0

    def selection(self):
        # Calculate fitness for each car
        for car in self.cars:
            car.fitness = car.calculate_fitness()

        # Update best fitness
        current_best = max(car.fitness for car in self.cars)
        self.best_fitness = max(self.best_fitness, current_best)

        # Sort cars by fitness
        self.cars.sort(key=lambda x: x.fitness, reverse=True)

        # Keep top performers
        self.cars = self.cars[:self.population_size // 2]

    def crossover(self):
        offspring = []

        while len(offspring) < self.population_size:
            parent1 = choice(self.cars)
            parent2 = choice(self.cars)

            child = Car()

            crossover_point = randint(0, parent1.weights1.size)
            child.weights1 = parent1.weights1.copy()
            child.weights2 = parent1.weights2.copy()

            child.weights1.flat[crossover_point:] = parent2.weights1.flat[crossover_point:]
            child.weights2.flat[crossover_point:] = parent2.weights2.flat[crossover_point:]

            offspring.append(child)

        self.cars.extend(offspring)

    def mutation(self, mutation_rate=0.1):
        for car in self.cars:
            if random() < mutation_rate:
                car.weights1 += np.random.randn(*car.weights1.shape) * 0.1
                car.weights2 += np.random.randn(*car.weights2.shape) * 0.1

    def start_new_generation(self):
        self.selection()
        self.crossover()
        self.mutation()
        self.generation += 1
        self.generation_start_time = time.time()

        for car in self.cars:
            car.reset_position()

        print(
            f"Generation {self.generation}, Best Fitness: {self.cars[0].fitness:.2f}, Best Overall: {self.best_fitness:.2f}")

