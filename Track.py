import pygame
import math

# Colors
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
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

        # Calculate track directions at each segment
        self.segment_directions = self._calculate_segment_directions()

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

    def _calculate_segment_directions(self):
        directions = []
        for i in range(len(self.racing_line) - 1):
            p1 = self.racing_line[i]
            p2 = self.racing_line[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            angle = math.degrees(math.atan2(dy, dx))
            directions.append(angle)
        return directions

    def _calculate_segment_lengths(self):
        lengths = []
        for i in range(len(self.racing_line) - 1):
            p1 = self.racing_line[i]
            p2 = self.racing_line[i + 1]
            length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            lengths.append(length)
        return lengths

    def get_progress_on_track(self, x, y):
        min_dist = float('inf')
        closest_segment = 0
        progress_in_segment = 0
        segment_direction = 0

        for i in range(len(self.racing_line) - 1):
            p1 = self.racing_line[i]
            p2 = self.racing_line[i + 1]

            segment_vec = (p2[0] - p1[0], p2[1] - p1[1])
            point_vec = (x - p1[0], y - p1[1])
            segment_length = self.segment_lengths[i]

            dot_product = (point_vec[0] * segment_vec[0] + point_vec[1] * segment_vec[1])
            t = max(0, min(1, dot_product / (segment_length * segment_length)))

            closest_x = p1[0] + t * segment_vec[0]
            closest_y = p1[1] + t * segment_vec[1]

            dist = math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)

            if dist < min_dist:
                min_dist = dist
                closest_segment = i
                progress_in_segment = t
                segment_direction = self.segment_directions[i]

        progress = (sum(self.segment_lengths[:closest_segment]) +
                    progress_in_segment * self.segment_lengths[closest_segment]) / self.total_track_length

        return progress, min_dist, segment_direction

    def draw(self, screen):
        # Draw track boundaries
        pygame.draw.polygon(screen, WHITE, self.outer_points, 2)
        pygame.draw.polygon(screen, WHITE, self.inner_points, 2)

        # Draw racing line with distinct points
        for i in range(len(self.racing_line) - 1):
            pygame.draw.line(screen, GRAY, self.racing_line[i], self.racing_line[i + 1], 2)
            pygame.draw.circle(screen, YELLOW, (int(self.racing_line[i][0]), int(self.racing_line[i][1])), 3)
