import pygame
import numpy as np
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CAR_WIDTH = 20
CAR_HEIGHT = 10
FPS = 60
NUM_CARS = 20
NUM_SENSORS = 5
MUTATION_RATE = 0.1
MAX_SPEED = 4
GENERATION_TIME = 10  # Seconds per generation
START_LINE = (150, 500)  # Cars start at the bottom-left
WAYPOINTS = [
    (150, 500),
    (650, 500),
    (650, 300),
    (150, 300),
    (150, 100),
    (650, 100),
]  # Waypoints defining the track

# Screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Self-Driving Car Simulation with Track")

# Track Surface
track = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
track.fill((255, 255, 255))  # White background
pygame.draw.rect(track, (0, 0, 0), (50, 50, 700, 500), width=10)  # Outer border


class Car:
    def __init__(self):
        self.x, self.y = START_LINE  # Start position
        self.angle = 0  # Initial angle
        self.speed = 2  # Small initial speed
        self.brain = np.random.uniform(-1, 1, (NUM_SENSORS, 2))  # Neural network weights
        self.fitness = 0
        self.alive = True
        self.sensors = [0] * NUM_SENSORS
        self.next_waypoint = 1  # Start aiming for the second waypoint
        self.waypoints_passed = 0  # Count of waypoints passed

    def update(self, track):
        if not self.alive:
            return

        # Update sensors
        self.update_sensors(track)

        # Use sensors to decide movement
        sensor_inputs = np.array(self.sensors) / 100.0  # Normalize sensor distances
        decision = np.dot(sensor_inputs, self.brain)  # Calculate turn and speed adjustments
        turn, acceleration = decision
        self.angle += turn * 0.1
        self.speed = min(MAX_SPEED, max(0, self.speed + acceleration * 0.1))

        # Update position
        self.x += self.speed * np.cos(self.angle)
        self.y += self.speed * np.sin(self.angle)

        # Collision detection
        if (
            self.x <= 50
            or self.x >= 750
            or self.y <= 50
            or self.y >= 550
            or track.get_at((int(self.x), int(self.y))) == (0, 0, 0)
        ):
            self.alive = False

        # Check if the car has reached the next waypoint
        target_x, target_y = WAYPOINTS[self.next_waypoint]
        distance_to_waypoint = np.hypot(self.x - target_x, self.y - target_y)

        if distance_to_waypoint < 30:  # If close enough to the waypoint
            self.next_waypoint += 1
            self.waypoints_passed += 1
            self.fitness += 100  # Reward for reaching a waypoint
            if self.next_waypoint >= len(WAYPOINTS):  # Loop waypoints
                self.next_waypoint = 0

        # Penalize for reversing direction or stalling
        if self.next_waypoint > 0:
            prev_x, prev_y = WAYPOINTS[self.next_waypoint - 1]
            dist_from_previous = np.hypot(self.x - prev_x, self.y - prev_y)
            if dist_from_previous < distance_to_waypoint:
                self.fitness -= 1  # Penalize for moving away from the waypoint

        # Reward for reducing distance to the next waypoint
        self.fitness += 1 / (distance_to_waypoint + 1)

    def update_sensors(self, track):
        for i, offset in enumerate(np.linspace(-np.pi / 2, np.pi / 2, NUM_SENSORS)):
            sensor_angle = self.angle + offset
            for dist in range(1, 100):
                sensor_x = int(self.x + dist * np.cos(sensor_angle))
                sensor_y = int(self.y + dist * np.sin(sensor_angle))
                if (
                    sensor_x <= 0
                    or sensor_x >= SCREEN_WIDTH
                    or sensor_y <= 0
                    or sensor_y >= SCREEN_HEIGHT
                    or track.get_at((sensor_x, sensor_y)) == (0, 0, 0)
                ):
                    self.sensors[i] = dist
                    break

    def draw(self):
        if not self.alive:
            return
        car_rect = pygame.Rect(
            self.x - CAR_WIDTH / 2, self.y - CAR_HEIGHT / 2, CAR_WIDTH, CAR_HEIGHT
        )
        pygame.draw.rect(screen, (255, 0, 0), car_rect)

        # Draw sensors
        for i, dist in enumerate(self.sensors):
            angle = self.angle + np.linspace(-np.pi / 2, np.pi / 2, NUM_SENSORS)[i]
            end_x = self.x + dist * np.cos(angle)
            end_y = self.y + dist * np.sin(angle)
            pygame.draw.line(screen, (0, 255, 0), (self.x, self.y), (end_x, end_y), 1)

        # Draw line to next waypoint
        if self.alive:
            target_x, target_y = WAYPOINTS[self.next_waypoint]
            pygame.draw.line(screen, (0, 0, 255), (self.x, self.y), (target_x, target_y), 2)


class GeneticAlgorithm:
    def __init__(self):
        self.population = [Car() for _ in range(NUM_CARS)]
        self.generation = 1

    def evolve(self):
        self.generation += 1
        fitness = [car.fitness for car in self.population]

        # Debugging: Print average and best fitness
        print(f"Generation {self.generation}: Best fitness = {max(fitness)}, Average fitness = {sum(fitness) / len(fitness)}")

        # Normalize fitness
        min_fitness = min(fitness)
        fitness = [f - min_fitness + 1 for f in fitness]  # Ensure all values are positive

        total_fitness = sum(fitness)
        probabilities = [f / total_fitness for f in fitness]

        # Select parents
        parents = random.choices(self.population, probabilities, k=NUM_CARS)

        # Crossover and mutation
        new_population = []
        for _ in range(NUM_CARS // 2):
            p1, p2 = random.sample(parents, 2)
            child1, child2 = self.crossover(p1, p2)
            new_population.append(self.mutate(child1))
            new_population.append(self.mutate(child2))

        self.population = new_population

    def crossover(self, p1, p2):
        child1 = Car()
        child2 = Car()
        crossover_point = random.randint(0, NUM_SENSORS - 1)
        child1.brain[:crossover_point] = p1.brain[:crossover_point]
        child1.brain[crossover_point:] = p2.brain[crossover_point:]
        child2.brain[:crossover_point] = p2.brain[:crossover_point]
        child2.brain[crossover_point:] = p1.brain[crossover_point:]
        return child1, child2

    def mutate(self, car):
        if random.random() < MUTATION_RATE:
            car.brain += np.random.uniform(-0.1, 0.1, car.brain.shape)
        return car


def main():
    clock = pygame.time.Clock()
    ga = GeneticAlgorithm()
    running = True
    start_time = pygame.time.get_ticks()

    while running:
        screen.blit(track, (0, 0))

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

        # Check if all cars are dead or generation time is up
        if all(not car.alive for car in ga.population) or (
            pygame.time.get_ticks() - start_time > GENERATION_TIME * 1000
        ):
            ga.evolve()
            start_time = pygame.time.get_ticks()

        # Display generation info
        font = pygame.font.SysFont(None, 36)
        gen_text = font.render(f"Generation: {ga.generation}", True, (0, 0, 0))
        screen.blit(gen_text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
