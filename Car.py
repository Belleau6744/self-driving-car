import pygame
import numpy as np
import math

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CAR_WIDTH = 20
CAR_HEIGHT = 10
FPS = 60
NUM_CARS = 30
NUM_SENSORS = 5
MUTATION_RATE = 0.01
MAX_SPEED = 30
GENERATION_TIME = 3  # Seconds per generation
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
    """Class representing a self-driving car in the simulation."""

    def __init__(self):
        self.brain = np.random.uniform(-1, 1, (NUM_SENSORS + 1, 2))  # Include the track sensor
        self.reset()

    def reset(self):
        """Reset the car's position, state, and properties."""
        self.x, self.y = START_LINE
        self.angle = 0
        self.speed = 2
        self.fitness = 0
        self.alive = True  # Ensure alive is set during reset
        self.sensors = [0] * NUM_SENSORS  # Normal sensors
        self.track_distance = 0  # Separate track distance sensor
        self.next_waypoint = 1
        self.waypoints_passed = 0
        self.prev_distance_to_waypoint = None

    def update_sensors(self, track):
        """Update sensor distances based on proximity to track boundaries."""
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
                        or track.get_at((sensor_x, sensor_y)) == (0, 0, 0)  # Hit boundary
                ):
                    self.sensors[i] = dist
                    break

        # Update track distance as a separate sensor
        self.track_distance = min(100, self.distance_to_track_line())  # Cap at max 100

    def update(self, track):
        """Update the car's state based on its neural network and surroundings."""
        if not self.alive:
            return

        # Update sensor readings
        self.update_sensors(track)

        # Prepare input for the neural network (sensors + track distance)
        sensor_inputs = np.array(self.sensors + [self.track_distance]) / 100.0  # Normalize sensor data

        # Neural network decision
        decision = np.dot(sensor_inputs, self.brain)
        turn, acceleration = decision
        self.angle += turn * 0.1  # Adjust direction
        self.speed = min(MAX_SPEED, max(0, self.speed + acceleration * 0.1))  # Adjust speed

        # Move the car
        self.x += self.speed * np.cos(self.angle)
        self.y += self.speed * np.sin(self.angle)

        # Check boundaries
        if self.x < 0 or self.x >= SCREEN_WIDTH or self.y < 0 or self.y >= SCREEN_HEIGHT:
            self.alive = False
            self.fitness -= 50  # Penalize for going out of bounds
            return

        if track.get_at((int(self.x), int(self.y))) == (255, 255, 255):  # Off track
            self.fitness -= 0.5  # Penalize slightly
        else:
            self.fitness += 0.5  # Reward for staying on the track

        # Handle waypoint progress
        self.handle_waypoint_progress()

        # Reward for staying close to the track line
        if self.track_distance < 30:
            self.fitness += (30 - self.track_distance) * 0.1  # Proximity reward
        else:
            self.fitness -= 0.1  # Penalty for being far

    def distance_to_track_line(self):
        """Calculate the minimum distance between the car and the track line."""
        min_distance = float('inf')
        for i in range(1, len(WAYPOINTS)):
            x1, y1 = WAYPOINTS[i - 1]
            x2, y2 = WAYPOINTS[i]
            # Point-to-line distance formula
            num = abs((y2 - y1) * self.x - (x2 - x1) * self.y + x2 * y1 - y2 * x1)
            den = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distance = num / den
            min_distance = min(min_distance, distance)
        return min_distance

    def handle_waypoint_progress(self):
        """Update fitness and waypoints if the car gets closer or reaches a waypoint."""
        target_x, target_y = WAYPOINTS[self.next_waypoint]
        distance_to_waypoint = np.hypot(self.x - target_x, self.y - target_y)

        if distance_to_waypoint < 30:  # Reached waypoint
            self.next_waypoint = (self.next_waypoint + 1) % len(WAYPOINTS)  # Loop waypoints
            self.waypoints_passed += 1
            self.fitness += 200  # Big reward for reaching waypoint
            self.prev_distance_to_waypoint = None
            return

        # Reward progress toward the waypoint
        if self.prev_distance_to_waypoint is not None:
            progress = self.prev_distance_to_waypoint - distance_to_waypoint
            self.fitness += max(0, progress * 10)  # Progress reward
        self.prev_distance_to_waypoint = distance_to_waypoint

    def draw(self):
        """Render the car and its sensors on the screen."""
        if not self.alive:
            return
        car_rect = pygame.Rect(
            self.x - CAR_WIDTH / 2, self.y - CAR_HEIGHT / 2, CAR_WIDTH, CAR_HEIGHT
        )
        pygame.draw.rect(screen, (255, 0, 0), car_rect)

        # Draw sensor lines
        for i, dist in enumerate(self.sensors[:NUM_SENSORS]):
            angle = self.angle + np.linspace(-np.pi / 2, np.pi / 2, NUM_SENSORS)[i]
            end_x = self.x + dist * np.cos(angle)
            end_y = self.y + dist * np.sin(angle)
            pygame.draw.line(screen, (0, 255, 0), (self.x, self.y), (end_x, end_y), 1)

        # Draw line to next waypoint
        if self.alive:
            target_x, target_y = WAYPOINTS[self.next_waypoint]
            pygame.draw.line(screen, (0, 0, 255), (self.x, self.y), (target_x, target_y), 2)
