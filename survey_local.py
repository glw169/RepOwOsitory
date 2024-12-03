import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from queue import Queue
from queue import PriorityQueue
import random
import math

from discretemap import *

# Parameters
grid_size = (100, 100)  # Size of the map
lidar_range = 15  # LiDAR scan range
robot_pos = [10, 10]  # Starting position of the robot
abs_robot_pos = robot_pos
frames = 500  # Maximum simulation steps

abs_path = []
visible_path = []

colors = ['black', 'blue', 'lightgray', 'white']
cmap = ListedColormap(colors)

dmap = DiscreteMap(sys.argv[1], 5)
# Expand the obstacles by half the obstacle's radius (radius = 0.35)
#dmap.expand_obstacles(0.175)

grid_size = (dmap.grid_width, dmap.grid_height)
local_map = np.full((grid_size[0]*3, grid_size[1]*3), -1)
#local_map_path = np.zeros((grid_size[0]*3, grid_size[1]*3))
#abs_map_path = np.zeros(grid_size)
uncertainty_padding = 20
local_map_plan = np.full((grid_size[0]+uncertainty_padding*2, grid_size[1]+uncertainty_padding*2), -1)

# Initialize environment and visible map
environment = np.zeros(grid_size)  # Ground truth map (0 = free space, 1 = obstacle)
visible_map = np.full(grid_size, -1)  # Visible map (-1 = unexplored)

#print(dmap.start)
robot_pos = (dmap.start[0], dmap.start[1])
abs_robot_pos = robot_pos

#mask = np.array(dmap.occupied).transpose()
#environment[mask] = 1

x_coords = [c[0] for c in dmap.occupied]
y_coords = [c[1] for c in dmap.occupied]
environment[x_coords, y_coords] = 1

# Add a solid vertical wall dividing the space
#wall_x = grid_size[0] // 2
#environment[wall_x, :] = 1  # Impassable wall

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
def update_map(map, lidar_scan):
    global environment

    for x, y in lidar_scan:
        if map[x, y] == -1:
            map[x, y] = 0  # Mark as free space
        
        if environment[x, y] == 1:
            map[x, y] = 1  # Mark as obstacle

    return map

def weighted_astar(visible_map, start, goal):
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
                if visible_map[nx, ny] != 1:  # Free space
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

# Path planning using BFS with debugging
def bfs(map, start):
    rows, cols = map.shape
    queue = Queue()
    queue.put(start)
    visited = set()
    parent = {tuple(start): None}
    print(f"Starting BFS from: {start}")
    
    while not queue.empty():
        current = queue.get()
        visited.add(tuple(current))
        
        if map[current[0], current[1]] == -1:
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
                map[nx, ny] != 1):  # Avoid obstacles
                queue.put([nx, ny])
                visited.add((nx, ny))
                parent[(nx, ny)] = current  # Track parent node
    
    print("No unexplored areas accessible.")
    return None  # No path found

"""
def store_paths(start, abs_start, end):
    global local_map_path
    global abs_map_path

    delta_x, delta_y = end[0] - start[0], end[1] - start[1]
    center = (grid_size[0] + grid_size[0]//2, grid_size[1] + grid_size[1]//2)
    local_map_path[center] = 2
    local_map_path = np.roll(local_map, (delta_x*-1, delta_y*-1), axis=(0,1))
    local_map_path[center] = 2

    abs_map_path[abs_start] = 2
    abs_map_path[end] = 2
"""

def update_position(start, abs_start, end, motion_error=0):
    global local_map
    global dmap
    delta_x, delta_y = end[0] - start[0], end[1] - start[1]

    local_map = np.roll(local_map, (delta_x*-1, delta_y*-1), axis=(0,1))

    if motion_error == 0:
        return end, end
    
    if random.random() > motion_error:
        return end, (abs_start[0] + delta_x, abs_start[1] + delta_y)
    
    if delta_x != 0 and delta_y == 0:
        potential_moves = [(delta_x, 0), (delta_x, 1), (delta_x, -1)]
        potential_moves = [move for move in potential_moves if (abs_start[0] + move[0], abs_start[1] + move[1]) not in dmap.occupied]

        delta = random.choice(potential_moves)
        return end, (abs_start[0] + delta[0], abs_start[1] + delta[1])
    
    if delta_y != 0 and delta_x == 0:
        potential_moves = [(0, delta_y), (1, delta_y), (-1, delta_y)]
        potential_moves = [move for move in potential_moves if (abs_start[0] + move[0], abs_start[1] + move[1]) not in dmap.occupied]

        delta = random.choice(potential_moves)
        return end, (abs_start[0] + delta[0], abs_start[1] + delta[1])
    
    potential_moves = [(delta_x, 0), (0, delta_y), (delta_x, delta_y)]
    potential_moves = [move for move in potential_moves if (abs_start[0] + move[0], abs_start[1] + move[1]) not in dmap.occupied]

    delta = random.choice(potential_moves)
    return end, (abs_start[0] + delta[0], abs_start[1] + delta[1])
    
     
def update_local_map(scan, robot_pos, abs_robot_pos, lidar_error=0):
    global grid_size
    global environment

    #center_scan = [(p[0]-robot_pos[0]+(grid_size[0] + grid_size[0]//2), p[1]-robot_pos[1]+(grid_size[1] + grid_size[1]//2)) for p in scan]

    #for i in range(len(center_scan)):
    #    local_map[center_scan[i]] = environment[scan[i]]

    offset_x = robot_pos[0] - abs_robot_pos[0]
    offset_y = robot_pos[1] - abs_robot_pos[1]
    for x, y in scan:
        local_map_plan[uncertainty_padding + offset_x + x][uncertainty_padding + offset_y + y] = environment[x][y]


def correct_motion_uncertainty(robot_pos, abs_pos, scan):

    calc_abs_pos = abs_pos #TODO: calc concentric center of lidar scan
    eliminated_error = abs(math.dist(calc_abs_pos, robot_pos))
    remaining_error = abs(math.dist(calc_abs_pos, abs_pos))
    print(f"Eliminated error: {eliminated_error}, Remaining error: {remaining_error}")
    return calc_abs_pos

path = []
# Simulation loop with debugging
for step in range(frames):
    print(f"Step {step}: Robot position: {robot_pos}, Absolute position: {abs_robot_pos}")
    
    # Simulate LiDAR scan
    lidar_scan = simulate_lidar(environment, abs_robot_pos, lidar_range)
    print(f"LiDAR scan detected {len(lidar_scan)} points.")

    #TODO: find concentric center of points from LiDAR scan to correct motion uncertainty
    #TODO: color motion uncertainty corrections
    #plt.imshow(local_map_plan, cmap=cmap)
    #plt.pause(5)
    # Update the visible map
    local_map_plan = update_map(local_map_plan, lidar_scan)

    #plt.imshow(local_map_plan, cmap=cmap)
    #plt.pause(20)
    
    # Plan path to nearest unexplored area
    path = bfs(local_map_plan, robot_pos)
    if path is None:
        print("No more unexplored areas accessible. Stopping simulation.")
        break
    if len(path) == 1:
        print("Crashed")
        break
    
    # Move the robot along the path
    robot_pos, abs_robot_pos = update_position(robot_pos, abs_robot_pos, path[1], motion_error=0.5)
    update_local_map(lidar_scan, robot_pos, abs_robot_pos)
    #store_paths(robot_pos, abs_robot_pos, path[1])
    #TODO: have an if run this every X steps to see a gradient of motion improvement
    robot_pos = correct_motion_uncertainty(robot_pos, abs_robot_pos, lidar_scan)
    #robot_pos = path[1]  # Move to the next step in the path
    print(f"Moving robot to: {robot_pos}")
    
    # Visualization
    #img = np.concatenate((visible_map, local_map[grid_size[0] : 2*grid_size[0], grid_size[1] : 2*grid_size[1]]), axis=1)
    #img = np.concatenate((img, local_map_plan), axis=1)

    #plt.imshow(img, cmap=cmap)
    #plt.scatter([abs_robot_pos[1], (grid_size[0]//2)+grid_size[0]], [abs_robot_pos[0], grid_size[1]//2], c='red', s=10, label='Abs Robot')
    #plt.scatter(robot_pos[1], robot_pos[0], c='purple', s=10, label='Robot')
    plt.imshow(local_map_plan, cmap=cmap)
    plt.scatter(abs_robot_pos[1]+uncertainty_padding, abs_robot_pos[0]+uncertainty_padding, c='purple', s=10)
    #plt.legend()
    plt.title(f"Step {step + 1}")
    plt.pause(0.1)
    plt.clf()

else:
    print("Maximum simulation step reached. Terminating")

    #x_coords = [c[0] for c in path]
    #y_coords = [c[1] for c in path]
    #visible_map[x_coords, y_coords] = 0.5
    #img = np.concatenate((visible_map, local_map[grid_size[0] : 2*grid_size[0], grid_size[1] : 2*grid_size[1]]), axis=1)

plt.imshow(local_map_plan, cmap=cmap)
plt.pause(5)


print("Simulation complete.")