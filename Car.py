import pygame
import numpy as np
import math
import time

# Constants
FPS = 60

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


class Car:
    def __init__(self):
        self.weights1 = np.random.randn(5, 8)
        self.weights2 = np.random.randn(8, 2)
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
        self.wrong_direction_time = 0
        self.backward_time = 0
        self.avg_racing_line_distance = 0
        self.racing_line_samples = 0

        # New time tracking variables
        self.lap_start_time = time.time()
        self.best_lap_time = float('inf')
        self.total_time = 0
        self.last_checkpoint_time = time.time()
        self.checkpoint_progress = 0


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

    def is_moving_in_wrong_direction(self, track_direction):
        # Get car's movement direction
        car_direction = self.angle

        # Calculate the difference between car direction and track direction
        angle_diff = (car_direction - track_direction) % 360
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # Consider the car to be going in the wrong direction if it's more than 90 degrees off
        return angle_diff > 90

    def update(self, track):
        if not self.alive:
            return

        current_time = time.time()

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
        self.speed = max(-2, min(10, self.speed))

        # Track backward movement
        if self.speed < 0:
            self.backward_time += 1
            if self.backward_time > FPS * 1.5:
                self.alive = False
                return
        else:
            self.backward_time = max(0, self.backward_time - 1)

        self.angle += self.rotation

        # Position update
        self.x += math.cos(math.radians(self.angle)) * self.speed
        self.y += math.sin(math.radians(self.angle)) * self.speed

        # Check if car is on track
        if not self.point_in_track(self.x, self.y, track):
            self.alive = False
            return

        # Update progress and check direction
        progress, distance_from_racing_line, track_direction = track.get_progress_on_track(self.x, self.y)

        # Update average distance from racing line
        self.avg_racing_line_distance = (
                                                    self.avg_racing_line_distance * self.racing_line_samples + distance_from_racing_line) / (
                                                    self.racing_line_samples + 1)
        self.racing_line_samples += 1

        # Check progress checkpoints for timeout prevention
        progress_checkpoint = int(progress * 10)  # Split track into 10 segments
        if progress_checkpoint != self.checkpoint_progress:
            self.checkpoint_progress = progress_checkpoint
            self.last_checkpoint_time = current_time
        elif current_time - self.last_checkpoint_time > 5:  # 5 seconds without reaching new checkpoint
            self.alive = False
            return

        # Check if moving in wrong direction
        if self.is_moving_in_wrong_direction(track_direction):
            self.wrong_direction_time += 1
            if self.wrong_direction_time > FPS * 2:
                self.alive = False
                return
        else:
            self.wrong_direction_time = 0

        # Detect if car completed a lap
        if progress < 0.1 and self.last_progress > 0.9:
            lap_time = current_time - self.lap_start_time
            self.best_lap_time = min(self.best_lap_time, lap_time)
            self.lap_start_time = current_time
            self.laps_completed += 1

        # Update best progress
        if progress > self.best_progress:
            self.best_progress = progress

        # Check if car is stuck
        if abs(progress - self.last_progress) < 0.001:
            self.stuck_time += 1
            if self.stuck_time > FPS * 3:
                self.alive = False
        else:
            self.stuck_time = 0

        self.last_progress = progress
        self.total_time = current_time - self.lap_start_time

        # Penalize being far from racing line
        if distance_from_racing_line > 100:
            self.alive = False
        elif distance_from_racing_line > 50:
            self.speed *= 0.98

    def calculate_fitness(self):
        # Base progress and lap completion rewards
        progress_fitness = self.best_progress * 500  # Reduced base progress reward

        # Lap time based rewards
        if self.laps_completed > 0:
            # Lower best_lap_time means higher fitness
            lap_time_fitness = 3000 * (1 / max(1, self.best_lap_time))
            # Additional bonus for completing multiple laps quickly
            multi_lap_bonus = self.laps_completed * 1000 * (1 / max(1, self.total_time / self.laps_completed))
        else:
            lap_time_fitness = 0
            multi_lap_bonus = 0

        # Speed bonus (only if moving forward in the right direction)
        speed_bonus = max(0, self.speed) * 20 if self.wrong_direction_time == 0 and self.backward_time == 0 else 0

        # Penalties
        direction_penalty = self.wrong_direction_time * 5
        backward_penalty = self.backward_time * 8

        # Racing line adherence (reduced impact compared to speed/lap time)
        max_preferred_distance = 50
        racing_line_reward = max(0, 200 * (1 - (self.avg_racing_line_distance / max_preferred_distance)))

        fitness = (progress_fitness +
                   lap_time_fitness +
                   multi_lap_bonus +
                   speed_bonus -
                   direction_penalty -
                   backward_penalty +
                   racing_line_reward)
        print(f"Fitness for car: {fitness}")  # Debug output
        return fitness

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

        # Color based on distance from racing line and movement
        if self.backward_time > 0:
            car_color = RED
        elif self.wrong_direction_time > 0:
            car_color = YELLOW
        else:
            # Gradient from green to white based on distance from racing line
            distance_factor = min(1, self.avg_racing_line_distance / 50)
            car_color = (
                int(GREEN[0] + (255 - GREEN[0]) * distance_factor),
                int(GREEN[1] + (255 - GREEN[1]) * distance_factor),
                int(GREEN[2] + (255 - GREEN[2]) * distance_factor)
            )

        pygame.draw.polygon(screen, car_color, car_points)

        # Draw sensors
        for i, reading in enumerate(self.sensor_readings):
            angle = self.angle - 90 + i * 45
            end_x = self.x + math.cos(math.radians(angle)) * reading
            end_y = self.y + math.sin(math.radians(angle)) * reading
            pygame.draw.line(screen, BLUE, (self.x, self.y), (end_x, end_y), 1)

