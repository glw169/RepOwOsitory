import numpy as np
import matplotlib.pyplot as plt
from queue import Queue

from discretemap import *

# Parameters
grid_size = (100, 100)  # Size of the map
lidar_range = 15        # LiDAR scan range
robot_pos = [10, 10]    # Starting position of the robot
frames = 1000           # Maximum simulation steps

# Add a solid vertical wall dividing the space
# wall_x = grid_size[0] // 2
# environment[wall_x, :] = 1  # Impassable wall

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

# Path planning using BFS with debugging
def bfs(visible_map, start):
    rows, cols = visible_map.shape
    queue = Queue()
    queue.put(start)
    visited = set()
    parent = {tuple(start): None}
    print(f"Starting BFS from: {start}")
    
    while not queue.empty():
        current = queue.get()
        visited.add(tuple(current))
        
        if visible_map[current[0], current[1]] == -1:
            # Found unexplored area
            print(f"Unexplored area found at: {current}")
            path = []
            while current is not None:
                path.append(current)
                current = parent[tuple(current)]
            return path[::-1]  # Return reversed path
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if (0 <= nx < rows and 0 <= ny < cols and
                (nx, ny) not in visited and
                visible_map[nx, ny] != 1):  # Avoid obstacles
                queue.put([nx, ny])
                visited.add((nx, ny))
                parent[(nx, ny)] = current  # Track parent node
    
    print("No unexplored areas accessible.")
    return None  # No path found


dmap = DiscreteMap(sys.argv[1], 5)
# Expand the obstacles by half the obstacle's radius (radius = 0.35)
# dmap.expand_obstacles(0.175)

grid_size = (dmap.grid_width, dmap.grid_height)
# Initialize environment and visible map
environment = np.zeros(grid_size)  # Ground truth map (0 = free space, 1 = obstacle)
visible_map = np.full(grid_size, -1)  # Visible map (-1 = unexplored)

# print(dmap.start)
robot_pos = dmap.start

# mask = np.array(dmap.occupied).transpose()
# environment[mask] = 1

x_coords = [c[0] for c in dmap.occupied]
y_coords = [c[1] for c in dmap.occupied]
environment[x_coords, y_coords] = 1

# for p in dmap.occupied:
#    print(p)

path = []
# Simulation loop with debugging
for step in range(frames):
    print(f"Step {step}: Robot position: {robot_pos}")
    
    # Simulate LiDAR scan
    lidar_scan = simulate_lidar(environment, robot_pos, lidar_range)
    print(f"LiDAR scan detected {len(lidar_scan)} points.")
    
    # Update the visible map
    update_map(visible_map, lidar_scan)
    
    # Plan path to nearest unexplored area
    path = bfs(visible_map, robot_pos)
    if path is None:
        print("No more unexplored areas accessible. Stopping simulation.")
        break
    
    # Move the robot along the path
    robot_pos = path[1]  # Move to the next step in the path
    print(f"Moving robot to: {robot_pos}")
    
    # Visualization
    plt.imshow(visible_map, cmap='gray')
    plt.scatter(robot_pos[1], robot_pos[0], c='red', s=10, label='Robot')
    plt.legend()
    plt.title(f"Step {step + 1}")
    plt.pause(0.001)
    plt.clf()
else:
    print("Maximum simulation step reached. Terminating")

    x_coords = [c[0] for c in path]
    y_coords = [c[1] for c in path]
    visible_map[x_coords, y_coords] = 0.5

plt.imshow(visible_map, cmap='gray')
plt.pause(10)

print("Simulation complete.")