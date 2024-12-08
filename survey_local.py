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
frames = 4000  # Maximum simulation steps

abs_path = []
visible_path = []

colors = ['black', 'lightgray', 'white', 'blue']
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
robot_pos = (dmap.start[0]+uncertainty_padding, dmap.start[1]+uncertainty_padding)
abs_robot_pos = robot_pos

#mask = np.array(dmap.occupied).transpose()
#environment[mask] = 1

x_coords = [c[0] for c in dmap.occupied]
y_coords = [c[1] for c in dmap.occupied]
environment[x_coords, y_coords] = 1

#x_coords = [c[0] for c in dmap.occupied_adjacent]
#y_coords = [c[1] for c in dmap.occupied_adjacent]
#environment[x_coords, y_coords] = 0.5

# Add a solid vertical wall dividing the space
#wall_x = grid_size[0] // 2
#environment[wall_x, :] = 1  # Impassable wall

# Simulate LiDAR scan within a circular radius
def simulate_lidar(environment, robot_pos, lidar_range):
    rows, cols = environment.shape
    lidar_scan = []
    for angle in np.linspace(0, 2 * np.pi, 360):  # Simulate 360-degree scan
        for r in range(1, lidar_range + 1):
            x = int(robot_pos[0] + r * np.cos(angle)) - uncertainty_padding
            y = int(robot_pos[1] + r * np.sin(angle)) - uncertainty_padding
            if 0 <= x < rows and 0 <= y < cols:
                if environment[x, y] == 1:  # Obstacle detected
                    lidar_scan.append((x, y))
                    break
                lidar_scan.append((x, y))

    r_x, r_y = robot_pos[0]-uncertainty_padding, robot_pos[1]-uncertainty_padding
    adjacent = [(r_x-1,r_y+1), (r_x,r_y+1), (r_x+1,r_y+1),
                (r_x-1,r_y  ),              (r_x+1,r_y  ),
                (r_x-1,r_y-1), (r_x,r_y-1), (r_x+1,r_y-1)]
    for x, y in adjacent:
        if 0 <= x < rows and 0 <= y < cols:
            if environment[x, y] == 1:  # Obstacle detected
                lidar_scan.append((x, y))

    return lidar_scan

# Update the visible map based on LiDAR data
def update_map(map, lidar_scan):
    global environment
    global visible_map

    for x, y in lidar_scan:
        if visible_map[x, y] == -1:
            visible_map[x, y] = 0  # Mark as free space
            map[x+uncertainty_padding, y+uncertainty_padding] = 0
        
        if environment[x, y] == 1:
            visible_map[x, y] = 1  # Mark as obstacle
            map[x+uncertainty_padding, y+uncertainty_padding] = 1

    return map

def find_nearest_frontier(map, robot_pos):
    rows, cols = map.shape
    queue = PriorityQueue()
    queue.put((0, (int(robot_pos[0]), int(robot_pos[1]))))
    visited = set()
    visited.add((int(robot_pos[0]), int(robot_pos[1])))

    while not queue.empty():
        distance, (x, y) = queue.get()

        if map[x, y] == -1:  # Unexplored space

            """
            if x > grid_size[0] + uncertainty_padding:
                x = x-1
            if x < uncertainty_padding:
                x = x+1
            if y > grid_size[1] + uncertainty_padding:
                y = y-1
            if y < uncertainty_padding:
                y = y+1
            """
                
            return (x, y)

        # Explore neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                if map[nx, ny] != 1 and map[nx, ny] != 2:  # Avoid walls and buffers
                    queue.put((distance + 1, (nx, ny)))
                    visited.add((nx, ny))
    return None  # No frontier found

def euclid_dist(start, end):
    return (end[0] - start[0])**2 + (end[1] - start[1])**2

def weighted_astar(map, start, goal):
    rows, cols = map.shape
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
                if map[nx, ny] != 1:  # Free space
                    new_cost = cost_so_far[current] + penalties[(nx, ny)] + euclid_dist((nx, ny), goal)#1
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

        if len(potential_moves) != 0:
            delta = random.choice(potential_moves)
        else:
            delta = (0,0)
        return end, (abs_start[0] + delta[0], abs_start[1] + delta[1])
    
    if delta_y != 0 and delta_x == 0:
        potential_moves = [(0, delta_y), (1, delta_y), (-1, delta_y)]
        potential_moves = [move for move in potential_moves if (abs_start[0] + move[0], abs_start[1] + move[1]) not in dmap.occupied]

        if len(potential_moves) != 0:
            delta = random.choice(potential_moves)
        else:
            delta = (0,0)
        return end, (abs_start[0] + delta[0], abs_start[1] + delta[1])
    
    potential_moves = [(delta_x, 0), (0, delta_y), (delta_x, delta_y)]
    potential_moves = [move for move in potential_moves if (abs_start[0] + move[0], abs_start[1] + move[1]) not in dmap.occupied]

    if len(potential_moves) != 0:
        delta = random.choice(potential_moves)
    else:
        delta = (0,0)

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

def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def convex_hull(points):
    # Sort the points lexicographically (tuples are compared lexicographically in Python)
    points = sorted(points)

    # Build the lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build the upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove the last point of each half because it's repeated at the beginning of the other half
    return lower[:-1] + upper[:-1]


def correct_motion_uncertainty(robot_pos, abs_pos, scan):

    #hull = np.array(convex_hull(scan))
    #hull[:, 0] += uncertainty_padding
    #hull[:, 1] += uncertainty_padding

    calc_abs_pos = abs_pos#np.mean(hull, axis=0, dtype=int) #abs_pos #TODO: calc concentric center of lidar scan
    eliminated_error = abs(math.dist(calc_abs_pos, robot_pos))
    remaining_error = abs(math.dist(calc_abs_pos, abs_pos))
    #print(f"Eliminated error: {eliminated_error}, Remaining error: {remaining_error}")
    return calc_abs_pos

class Moves():
    def __init__(self, rolling=False) -> None:
        self.moves = []
        self.new_trail = True
        self.rolling = rolling
    
    def add_move(self, m):
        if self.new_trail and self.rolling:
            self.moves = []

        self.new_trail = False
        if len(self.moves) < 10 or not self.rolling:
            self.moves.append(m)
        else:
            self.moves = self.moves[1:] + self.moves[:1]
            self.moves.pop()

    def get_recent_move(self):
        if self.new_trail == False:
            self.new_trail = True

        if len(self.moves):
            return self.moves.pop(0)
        else:
            return None
        
def compute_wall_penalties(factor):

    penalties = {}

    for map_x in range(uncertainty_padding*2+grid_size[0]):
        for map_y in range(uncertainty_padding*2+grid_size[1]):
        
            x, y = map_x - uncertainty_padding, map_y - uncertainty_padding
            adjacent = [(x-1,y+1), (x,y+1), (x+1,y+1),
                        (x-1,y  ),          (x+1,y  ),
                        (x-1,y-1), (x,y-1), (x+1,y-1)]
            
            adj_walls = [adj for adj in adjacent if adj in dmap.occupied]
            penalties[(map_x, map_y)] = (len(adj_walls)*factor)**2
    
    return penalties


timeout = 10
timeout_cnt = 0
previous_pos = Moves()
path = []
dmap.occupied = set(dmap.occupied)
penalties = compute_wall_penalties(10)

starting_map = environment.copy()
blank = np.full((grid_size[0]+uncertainty_padding*2, grid_size[1]+uncertainty_padding*2), -1)

start_x = (blank.shape[0] - starting_map.shape[0]) // 2
start_y = (blank.shape[1] - starting_map.shape[1]) // 2

blank[start_x:start_x + starting_map.shape[0], start_y:start_y + starting_map.shape[1]] = starting_map
starting_map = blank.copy()

plt.imshow(starting_map, cmap=ListedColormap(['black', 'lightgray', 'white']))
plt.pause(20)

# Simulation loop with debugging
for step in range(frames):
    #print(f"Step {step}: Robot position: {robot_pos}, Absolute position: {abs_robot_pos}")
    
    # Simulate LiDAR scan
    lidar_scan = simulate_lidar(environment, abs_robot_pos, lidar_range)
    #print(f"LiDAR scan detected {len(lidar_scan)} points.")

    #TODO: find concentric center of points from LiDAR scan to correct motion uncertainty
    #TODO: color motion uncertainty corrections
    #plt.imshow(local_map_plan, cmap=cmap)
    #plt.pause(5)
    # Update the visible map
    local_map_plan = update_map(local_map_plan, lidar_scan)

    """
    path = bfs(local_map_plan, robot_pos)
    if path is None:
        print("No more unexplored areas accessible. Stopping simulation.")
        break
    if len(path) == 1:
        print("Crashed")
    robot_pos, abs_robot_pos = update_position(robot_pos, abs_robot_pos, path[1], motion_error=0.1)
    """

    
    target = find_nearest_frontier(local_map_plan, robot_pos)
    if target is None:
        print("Trying from pos:", robot_pos)
        robot_pos = previous_pos.get_recent_move()
        timeout_cnt += 1
        print("crash:", timeout_cnt)
        if timeout_cnt == timeout or robot_pos is None:
            print("Exploration complete.")
            final_map = visible_map.copy()
            break
        continue

    timeout_cnt = 0

    path = weighted_astar(local_map_plan, (int(robot_pos[0]), int(robot_pos[1])), target)

    if len(path) == 0:
        print("Trying from pos:", robot_pos)
        robot_pos = previous_pos.get_recent_move()
        timeout_cnt += 1
        print("crash:", timeout_cnt)
        if timeout_cnt == timeout or robot_pos is None:
            print("Exploration complete.")
            final_map = visible_map.copy()
            break
        continue
    
    # Move the robot along the path
    previous_pos.add_move((robot_pos[0], robot_pos[1]))
    robot_pos, abs_robot_pos = update_position(robot_pos, abs_robot_pos, path[0], motion_error=0.6)
    
    update_local_map(lidar_scan, robot_pos, abs_robot_pos)
    #store_paths(robot_pos, abs_robot_pos, path[1])
    #TODO: have an if run this every X steps to see a gradient of motion improvement
    robot_pos = correct_motion_uncertainty(robot_pos, abs_robot_pos, lidar_scan)
    #robot_pos = path[1]  # Move to the next step in the path
    #print(f"Moving robot to: {robot_pos}")
    
    # Visualization
    #img = np.concatenate((visible_map, local_map[grid_size[0] : 2*grid_size[0], grid_size[1] : 2*grid_size[1]]), axis=1)
    #img = np.concatenate((img, local_map_plan), axis=1)

    #plt.imshow(img, cmap=cmap)
    #plt.scatter([abs_robot_pos[1], (grid_size[0]//2)+grid_size[0]], [abs_robot_pos[0], grid_size[1]//2], c='red', s=10, label='Abs Robot')
    #plt.scatter(robot_pos[1], robot_pos[0], c='purple', s=10, label='Robot')
    local_plan_w_path = local_map_plan.copy()
    x_coords = [c[0] for c in path]
    y_coords = [c[1] for c in path]
    local_plan_w_path[x_coords, y_coords] = 2

    plt.imshow(local_plan_w_path, cmap=cmap)
    plt.scatter(robot_pos[1], robot_pos[0], c='purple', s=10)
    #plt.legend()
    plt.title(f"Step {step + 1}")
    plt.pause(0.01)
    plt.clf()

else:
    print("Maximum simulation step reached. Terminating")

    #x_coords = [c[0] for c in path]
    #y_coords = [c[1] for c in path]
    #visible_map[x_coords, y_coords] = 0.5
    #img = np.concatenate((visible_map, local_map[grid_size[0] : 2*grid_size[0], grid_size[1] : 2*grid_size[1]]), axis=1)

plt.imshow(local_map_plan, cmap=ListedColormap(['black', 'lightgray', 'white']))
plt.pause(5)

diffs = starting_map != local_map_plan
error_map = starting_map.copy()
error_map[diffs] = 2

#print(np.unique(starting_map))
errors = diffs.sum()
total = environment.shape[0] * environment.shape[1]
print("Mapping Error:", str(errors*100/total) + "%")

plt.imshow(error_map, cmap=ListedColormap(['black', 'lightgray', 'white', 'red']))
plt.pause(20)

print("Simulation complete.")