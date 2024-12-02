import numpy as np
import matplotlib.pyplot as plt
from discretemap import *
import sys
from queue import PriorityQueue
import random

# Parameters
frames = 1000  # Maximum simulation steps
lidar_range = 15  # LiDAR range in grid cells
angle_resolution = 60  # Reduced angle resolution for LiDAR
buffer_radius = 1  # Buffer zone radius around obstacles
buffer_penalty = 10  # Penalty for moving through buffer zones
unreachable_targets = set()  # Track unreachable frontiers
max_stuck_attempts = 5  # Max retries before random movement

# Load the map
if len(sys.argv) < 2:
    print("Usage: python survey.py <map_file>")
    sys.exit(1)

dmap = DiscreteMap(sys.argv[1], 5)
grid_size = (dmap.grid_width, dmap.grid_height)
environment = np.zeros(grid_size)  # Ground truth map
visible_map = np.full(grid_size, -1)  # -1: unexplored, 0: free, 1: wall, 2: buffer zone

# Populate obstacles in the environment map
x_coords = [c[0] for c in dmap.occupied]
y_coords = [c[1] for c in dmap.occupied]
environment[x_coords, y_coords] = 1

# Robot initial pose
robot_pose = np.array([dmap.start[0], dmap.start[1], 0.0])  # x, y, theta

# Inflate obstacles to create buffer zones around walls
def inflate_obstacles(map_data, radius):
    rows, cols = map_data.shape
    inflated = map_data.copy()
    for x in range(rows):
        for y in range(cols):
            if map_data[x, y] == 1:  # Wall detected
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and inflated[nx, ny] != 1:
                            inflated[nx, ny] = 2  # Mark as buffer zone
    return inflated

# Simulate LiDAR scan
def simulate_lidar(environment, robot_pos, lidar_range, angle_resolution):
    lidar_scan = []
    angles = np.linspace(0, 2 * np.pi, angle_resolution)  # Reduced resolution
    rows, cols = environment.shape

    for angle in angles:
        dx = np.cos(angle)
        dy = np.sin(angle)
        for r in range(1, lidar_range + 1):
            x = int(robot_pos[0] + r * dx)
            y = int(robot_pos[1] + r * dy)
            if 0 <= x < rows and 0 <= y < cols:
                if environment[x, y] == 1:  # Obstacle detected
                    lidar_scan.append((x, y))
                    break
                lidar_scan.append((x, y))
            else:
                break
    return lidar_scan

# Update visible map with LiDAR data
def update_map(visible_map, lidar_scan):
    lidar_scan = np.array(lidar_scan)
    rows, cols = visible_map.shape
    valid_points = (0 <= lidar_scan[:, 0]) & (lidar_scan[:, 0] < rows) & \
                   (0 <= lidar_scan[:, 1]) & (lidar_scan[:, 1] < cols)
    lidar_scan = lidar_scan[valid_points]

    visible_map[lidar_scan[:, 0], lidar_scan[:, 1]] = np.where(
        environment[lidar_scan[:, 0], lidar_scan[:, 1]] == 1, 1, 0
    )
    return inflate_obstacles(visible_map, buffer_radius)

# Weighted A* Pathfinding
def weighted_astar(visible_map, start, goal, buffer_penalty):
    rows, cols = visible_map.shape
    queue = PriorityQueue()
    queue.put((0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not queue.empty():
        _, current = queue.get()

        if current == goal:
            break

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                # Add penalty for buffer zones
                if visible_map[nx, ny] == 2:  # Buffer zone
                    new_cost = cost_so_far[current] + 1 + buffer_penalty
                elif visible_map[nx, ny] != 1:  # Free space
                    new_cost = cost_so_far[current] + 1
                else:
                    continue  # Skip walls

                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost + abs(nx - goal[0]) + abs(ny - goal[1])
                    queue.put((priority, (nx, ny)))
                    came_from[(nx, ny)] = current

    # Reconstruct path
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:  # No path found
            return None
    path.reverse()
    return path

# Find the nearest frontier
def find_nearest_frontier(visible_map, robot_pos):
    rows, cols = visible_map.shape
    queue = PriorityQueue()
    queue.put((0, (int(robot_pos[0]), int(robot_pos[1]))))
    visited = set()
    visited.add((int(robot_pos[0]), int(robot_pos[1])))

    while not queue.empty():
        distance, (x, y) = queue.get()

        if visible_map[x, y] == -1:  # Unexplored space
            return (x, y)

        # Explore neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                if visible_map[nx, ny] != 1 and visible_map[nx, ny] != 2:  # Avoid walls and buffers
                    queue.put((distance + 1, (nx, ny)))
                    visited.add((nx, ny))
    return None  # No frontier found

# Initialize the figure for side-by-side plots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Main SLAM loop
stuck_count = 0
for step in range(frames):
    print(f"Step {step}: Robot pose: {robot_pose}")
    lidar_scan = simulate_lidar(environment, robot_pose[:2], lidar_range, angle_resolution)
    visible_map = update_map(visible_map, lidar_scan)

    target = find_nearest_frontier(visible_map, robot_pose)
    if target is None:
        print("Exploration complete.")
        break

    print(f"Target position: {target}")
    path = weighted_astar(visible_map, (int(robot_pose[0]), int(robot_pose[1])), target, buffer_penalty)

    if path is None:
        print(f"Target {target} is unreachable. Marking as unreachable.")
        unreachable_targets.add(target)
        stuck_count += 1
        if stuck_count >= max_stuck_attempts:
            print("Stuck! Forcing random movement.")
            robot_pose[0] += random.choice([-1, 1])
            robot_pose[1] += random.choice([-1, 1])
            stuck_count = 0
        continue

    stuck_count = 0
    if len(path) > 1:
        robot_pose = np.array([path[1][0], path[1][1], robot_pose[2]])

    # Clear both axes
    axs[0].clear()
    axs[1].clear()

    # Left plot: Robot with LiDAR scans (Static View)
    axs[0].imshow(visible_map, cmap='gray', origin='lower')
    axs[0].scatter(robot_pose[1], robot_pose[0], c='red', s=10, label='Robot')

    # Draw LiDAR lines
    for (x, y) in lidar_scan:
        dx = y - robot_pose[1]
        dy = x - robot_pose[0]
        axs[0].arrow(robot_pose[1], robot_pose[0], dx, dy, head_width=1, color='cyan', alpha=0.6)

    axs[0].set_title("Robot with LiDAR Scans (Static)")
    axs[0].legend()

    # Right plot: Robot moving to the target (Centered on Robot)
    axs[1].imshow(visible_map, cmap='gray', origin='lower')
    axs[1].scatter(robot_pose[1], robot_pose[0], c='red', s=10, label='Robot')
    axs[1].scatter(target[1], target[0], c='blue', s=50, marker='x', label='Target')

    # Center the camera on the robot
    axs[1].set_xlim([robot_pose[1] - lidar_range, robot_pose[1] + lidar_range])
    axs[1].set_ylim([robot_pose[0] - lidar_range, robot_pose[0] + lidar_range])
    axs[1].set_title("Robot Moving to Target (Centered)")
    axs[1].legend()

    # Refresh plots
    plt.pause(0.001)

plt.show()
