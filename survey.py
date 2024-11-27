import numpy as np
import matplotlib.pyplot as plt
from queue import Queue

from discretemap import *

# Parameters
grid_size = (100, 100)  # Size of the map
lidar_range = 15        # LiDAR scan range
robot_pos = [10, 10]    # Starting position of the robot
frames = 1000           # Maximum simulation steps

# Simulate LiDAR scan within a circular radius
def simulate_lidar(environment, robot_pos, lidar_range):
    rows, cols = environment.shape
    lidar_scan = []
    for angle in np.linspace(0, 2 * np.pi, 360):  # Simulate 360-degree scan
        for r in range(1, lidar_range + 1):
            x = int(robot_pos[0] + r * np.cos(angle))
            y = int(robot_pos[1] + r * np.sin(angle))
            if 0 <= x < rows and 0 <= y < cols:
                if environment[x, y] == 1:  # Obstacle detected
                    lidar_scan.append((x, y))
                    break
                lidar_scan.append((x, y))
    return lidar_scan

# Update the visible map based on LiDAR data
def update_map(visible_map, lidar_scan):
    global environment
    for x, y in lidar_scan:
        if visible_map[x, y] == -1:
            visible_map[x, y] = 0  # Mark as free space
    for x, y in lidar_scan:
        if environment[x, y] == 1:
            visible_map[x, y] = 1  # Mark as obstacle

# Check if a cell is a frontier
def is_frontier_cell(visible_map, x, y):
    if visible_map[x, y] != -1:  # Not unexplored
        return False
    rows, cols = visible_map.shape
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and visible_map[nx, ny] == 0:
            return True  # Adjacent to free space
    return False

# Detect all frontier clusters
def detect_frontiers(visible_map):
    rows, cols = visible_map.shape
    visited = np.zeros_like(visible_map, dtype=bool)
    frontiers = []

    for x in range(rows):
        for y in range(cols):
            if is_frontier_cell(visible_map, x, y) and not visited[x, y]:
                # Start a new frontier cluster
                cluster = []
                queue = Queue()
                queue.put((x, y))
                visited[x, y] = True

                while not queue.empty():
                    cx, cy = queue.get()
                    cluster.append((cx, cy))

                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < rows and 0 <= ny < cols and
                                not visited[nx, ny] and is_frontier_cell(visible_map, nx, ny)):
                            queue.put((nx, ny))
                            visited[nx, ny] = True

                frontiers.append(cluster)
    return frontiers

# Select the nearest frontier cluster
def select_frontier(frontiers, robot_pos):
    best_frontier = None
    best_cost = float('inf')

    for frontier in frontiers:
        # Calculate cost as distance to the closest cell in the cluster
        cost = min(np.linalg.norm(np.array(robot_pos) - np.array(cell)) for cell in frontier)
        if cost < best_cost:
            best_cost = cost
            best_frontier = frontier

    return best_frontier

def bfs(visible_map, start, goal):
    rows, cols = visible_map.shape
    queue = Queue()
    queue.put(start)
    visited = set()
    parent = {tuple(start): None}
    while not queue.empty():
        current = queue.get()
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[tuple(current)]
            return path[::-1]

        visited.add(tuple(current))

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if (0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited and visible_map[nx, ny] != 1):  # Avoid obstacles
                queue.put([nx, ny])
                visited.add((nx, ny))
                parent[(nx, ny)] = current
    return None  # No path found

dmap = DiscreteMap(sys.argv[1], 5)
grid_size = (dmap.grid_width, dmap.grid_height)
environment = np.zeros(grid_size)
visible_map = np.full(grid_size, -1)
x_coords = [c[0] for c in dmap.occupied]
y_coords = [c[1] for c in dmap.occupied]
environment[x_coords, y_coords] = 1

for step in range(frames):
    print(f"Step {step}: Robot position: {robot_pos}")
    
    # Simulate LiDAR scan
    lidar_scan = simulate_lidar(environment, robot_pos, lidar_range)
    update_map(visible_map, lidar_scan)
    
    # Detect frontiers
    frontiers = detect_frontiers(visible_map)
    print(f"Frontiers: {frontiers}")
    if not frontiers:
        print("No frontiers found. Stopping simulation.")
        break

    # Select the best frontier
    selected_frontier = select_frontier(frontiers, robot_pos)
    if not selected_frontier:
        print("No accessible frontier. Stopping simulation.")
        break
    
    # Plan path to the selected frontier
    goal = selected_frontier[0]
    path = bfs(visible_map, robot_pos, goal)
    if not path:
        print("No path to frontier. Stopping simulation.")
        break

    # Move the robot
    robot_pos = path[1]
    print(f"Moving robot to: {robot_pos}")

    # Visualization
    plt.imshow(visible_map, cmap='gray')
    plt.scatter(robot_pos[1], robot_pos[0], c='red', s=10, label='Robot')
    plt.legend()
    plt.title(f"Step {step + 1}")
    plt.pause(0.001)
    plt.clf()

print("Simulation complete.")
