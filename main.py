import pygame
import numpy as np
import math
from random import random, randint, choice
import sys
import time

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60
GENERATION_TIME_LIMIT = 15  # Time limit in seconds for each generation

# Starting position for cars
START_X = 250
START_Y = 400
START_ANGLE = 0

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (147, 112, 219)
GRAY = (128, 128, 128)


class Track:
    def __init__(self):
        # Track boundaries - making it more oval-shaped
        self.outer_points = [
            (200, 200),  # Top left
            (600, 150),  # Top middle
            (1000, 200),  # Top right
            (1100, 400),  # Right middle
            (1000, 600),  # Bottom right
            (600, 650),  # Bottom middle
            (200, 600),  # Bottom left
            (100, 400),  # Left middle
        ]

        # Inner track points - maintaining track width
        self.inner_points = [
            (300, 300),  # Top left
            (600, 250),  # Top middle
            (900, 300),  # Top right
            (950, 400),  # Right middle
            (900, 500),  # Bottom right
            (600, 550),  # Bottom middle
            (300, 500),  # Bottom left
            (250, 400),  # Left middle
        ]

        # Generate racing line and calculate lengths
        self.racing_line = self._generate_racing_line()
        self.segment_lengths = self._calculate_segment_lengths()
        self.total_track_length = sum(self.segment_lengths)

    def _generate_racing_line(self):
        # Calculate middle points between outer and inner track boundaries
        middle_points = []

        # For each corner of the track, calculate the midpoint
        for i in range(len(self.outer_points)):
            outer_point = self.outer_points[i]
            inner_point = self.inner_points[i]

            # Calculate midpoint between outer and inner track
            mid_x = (outer_point[0] + inner_point[0]) / 2
            mid_y = (outer_point[1] + inner_point[1]) / 2
            middle_points.append((mid_x, mid_y))

        # Add additional points along the straight sections
        racing_line = []
        num_intermediate_points = 5  # Number of points to add between corners

        for i in range(len(middle_points)):
            p1 = middle_points[i]
            p2 = middle_points[(i + 1) % len(middle_points)]

            racing_line.append(p1)

            # Add intermediate points
            for j in range(1, num_intermediate_points):
                t = j / num_intermediate_points
                x = p1[0] + (p2[0] - p1[0]) * t
                y = p1[1] + (p2[1] - p1[1]) * t
                racing_line.append((x, y))

        # Close the loop
        racing_line.append(racing_line[0])
        return racing_line

    def _calculate_segment_lengths(self):
        lengths = []
        for i in range(len(self.racing_line) - 1):
            p1 = self.racing_line[i]
            p2 = self.racing_line[i + 1]
            length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            lengths.append(length)
        return lengths

    def get_progress_on_track(self, x, y):
        # Find closest point on racing line and calculate progress
        min_dist = float('inf')
        closest_segment = 0
        progress_in_segment = 0

        for i in range(len(self.racing_line) - 1):
            p1 = self.racing_line[i]
            p2 = self.racing_line[i + 1]

            # Calculate closest point on line segment
            segment_vec = (p2[0] - p1[0], p2[1] - p1[1])
            point_vec = (x - p1[0], y - p1[1])
            segment_length = self.segment_lengths[i]

            # Calculate projection
            dot_product = (point_vec[0] * segment_vec[0] + point_vec[1] * segment_vec[1])
            t = max(0, min(1, dot_product / (segment_length * segment_length)))

            # Find closest point
            closest_x = p1[0] + t * segment_vec[0]
            closest_y = p1[1] + t * segment_vec[1]

            dist = math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)

            if dist < min_dist:
                min_dist = dist
                closest_segment = i
                progress_in_segment = t

        # Calculate total progress (0 to 1)
        progress = (sum(self.segment_lengths[:closest_segment]) +
                    progress_in_segment * self.segment_lengths[closest_segment]) / self.total_track_length

        return progress, min_dist

    def draw(self, screen):
        # Draw track boundaries
        pygame.draw.polygon(screen, WHITE, self.outer_points, 2)
        pygame.draw.polygon(screen, WHITE, self.inner_points, 2)

        # Draw racing line with distinct points
        for i in range(len(self.racing_line) - 1):
            pygame.draw.line(screen, GRAY, self.racing_line[i], self.racing_line[i + 1], 2)
            pygame.draw.circle(screen, YELLOW, (int(self.racing_line[i][0]), int(self.racing_line[i][1])), 3)


class Car:
    def __init__(self):
        self.weights1 = np.random.randn(5, 8)  # 5 sensors, 8 hidden neurons
        self.weights2 = np.random.randn(8, 2)  # 8 hidden neurons, 2 outputs
        self.reset_position()

    def reset_position(self):
        self.x = START_X
        self.y = START_Y
        self.angle = START_ANGLE
        self.speed = 0
        self.acceleration = 0
        self.rotation = 0
        self.alive = True
        self.fitness = 0
        self.laps_completed = 0
        self.best_progress = 0
        self.sensor_readings = []
        self.last_progress = 0
        self.stuck_time = 0

    def get_sensor_readings(self, track):
        readings = []
        for i in range(5):
            angle = self.angle - 90 + i * 45
            x = self.x
            y = self.y

            while True:
                x += math.cos(math.radians(angle)) * 5
                y += math.sin(math.radians(angle)) * 5

                if not self.point_in_track(x, y, track):
                    distance = math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)
                    readings.append(distance)
                    break

        self.sensor_readings = readings
        return readings

    def point_in_track(self, x, y, track):
        def point_in_polygon(x, y, points):
            n = len(points)
            inside = False
            p1x, p1y = points[0]
            for i in range(n + 1):
                p2x, p2y = points[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            return inside

        return (point_in_polygon(x, y, track.outer_points) and
                not point_in_polygon(x, y, track.inner_points))

    def update(self, track):
        if not self.alive:
            return

        # Neural network control
        readings = self.get_sensor_readings(track)
        readings = np.array(readings) / 200

        hidden = np.tanh(np.dot(readings, self.weights1))
        output = np.tanh(np.dot(hidden, self.weights2))

        self.acceleration = output[0] * 0.5
        self.rotation = output[1] * 3

        # Physics update
        self.speed += self.acceleration
        self.speed *= 0.95  # Friction
        self.speed = max(-5, min(10, self.speed))  # Speed limits

        self.angle += self.rotation

        # Position update
        self.x += math.cos(math.radians(self.angle)) * self.speed
        self.y += math.sin(math.radians(self.angle)) * self.speed

        # Check if car is on track
        if not self.point_in_track(self.x, self.y, track):
            self.alive = False
            return

        # Update progress
        progress, distance_from_racing_line = track.get_progress_on_track(self.x, self.y)

        # Detect if car completed a lap
        if progress < 0.1 and self.last_progress > 0.9:
            self.laps_completed += 1

        # Update best progress
        if progress > self.best_progress:
            self.best_progress = progress

        # Check if car is stuck
        if abs(progress - self.last_progress) < 0.001:
            self.stuck_time += 1
            if self.stuck_time > FPS * 3:  # 3 seconds of no progress
                self.alive = False
        else:
            self.stuck_time = 0

        self.last_progress = progress

        # Penalize being far from racing line
        if distance_from_racing_line > 100:
            self.alive = False

    def calculate_fitness(self):
        # Base fitness from progress and laps
        progress_fitness = self.best_progress * 1000
        lap_fitness = self.laps_completed * 2000

        # Speed bonus
        speed_bonus = max(0, self.speed) * 10

        return progress_fitness + lap_fitness + speed_bonus

    def draw(self, screen):
        if not self.alive:
            return

        # Draw car
        car_points = [
            (self.x + math.cos(math.radians(self.angle)) * 20,
             self.y + math.sin(math.radians(self.angle)) * 20),
            (self.x + math.cos(math.radians(self.angle + 120)) * 10,
             self.y + math.sin(math.radians(self.angle + 120)) * 10),
            (self.x + math.cos(math.radians(self.angle + 240)) * 10,
             self.y + math.sin(math.radians(self.angle + 240)) * 10)
        ]
        pygame.draw.polygon(screen, GREEN if self.alive else RED, car_points)

        # Draw sensors
        for i, reading in enumerate(self.sensor_readings):
            angle = self.angle - 90 + i * 45
            end_x = self.x + math.cos(math.radians(angle)) * reading
            end_y = self.y + math.sin(math.radians(angle)) * reading
            pygame.draw.line(screen, BLUE, (self.x, self.y), (end_x, end_y), 1)


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
