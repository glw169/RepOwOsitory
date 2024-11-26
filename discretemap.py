#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageColor
import math
import random
import os.path
import sys
import pickle

def wrap_angle(angle):
    r_angle = angle
    while r_angle <= -math.pi:
        r_angle += 2*math.pi
    while r_angle >= math.pi:
        r_angle -= 2*math.pi
    return r_angle

class DiscreteMap():
    def __init__(self,world_file,d_step):
        self.d_step = d_step
        self.map_width = 1.0
        self.map_height = 1.0
        self.image_width = 1.0
        self.image_height = 1.0
        self.grid_width = 1
        self.grid_height = 1
        self.world_file = world_file
        
        self.occupied = []
        self.num_angles = 72

        self.world_dir, self.world_name = os.path.split(self.world_file)
        print('Created discrete map: ')
        print('World Dir: ', self.world_dir)
        print('World name: ', self.world_name[0:-6])
        self.parse_worldfile()
        self.load_image()
        self.discretize()

        print('Discretized Map with d_step ', self.d_step, 'x y product', self.grid_width, self.grid_height, self.grid_width*self.grid_height)
        self.meters_per_grid_cell = self.map_width / self.grid_width
        
    def map_angle_to_grid_angle(self,angle):
        return -angle

    def map_angle_to_discrete_angle(self, angle):
        w_angle = wrap_angle(angle)
        if w_angle < 0:
            w_angle += 2*math.pi
        a_step = 2*math.pi / self.num_angles
        d_angle = int(math.floor(w_angle / a_step))
        return d_angle

    def discrete_angle_to_map_angle(self, d_angle):
        a_step = 2*math.pi / self.num_angles
        return wrap_angle(d_angle * a_step)

    def draw_discrete_map(self,sname):
        draw = ImageDraw.Draw(self.im)
        self.discretize(draw)
        self.im.save(sname)
        del draw

    def draw_obstacle_map(self,sname):
        display_im = self.im.copy()
        draw = ImageDraw.Draw(display_im)
        self.draw_grid_obstacles(draw)
        display_im.save(sname)
        del draw
        del display_im

    def load_image(self):
        print('Loading image: ', self.image_file)
        self.im = Image.open(self.image_file)
        self.im2 = self.im.convert("L")
        self.image_width = self.im.size[0]
        self.image_height = self.im.size[1]

    def draw_grid_obstacles(self, draw):
        for p in self.occupied:
            self.draw_grid_rectangle(p[0],p[1],draw,"Green")
        gstart = self.map_to_grid(self.start)
        ggoal = self.map_to_grid(self.goal)
        self.draw_grid_rectangle(gstart[0], gstart[1], draw, "Blue")
        self.draw_grid_rectangle(ggoal[0], ggoal[1], draw, "Red")

    def discretize(self, draw=None):
        pix = self.im2.load()
        
        x = 0 # x location in image
        gx = 0 # x location on grid

        while x + self.d_step <= self.image_width:
            if draw is not None:
                draw.line([x,0,x,self.image_height],fill='Orange')
            y = 0 # y location in image
            gy = 0 # y location on grid
            while y + self.d_step <= self.image_height:
                if draw is not None: 
                    draw.line([0,y,self.image_width,y],fill='Orange')

                obstacle = False
                for px in range(x,x+self.d_step):
                    for py in range(y,y+self.d_step):
                        if px < self.image_width and py < self.image_height:
                            if pix[px,py] == 0:
                                obstacle = True
                                break
                    
                    if obstacle:
                        break

                if obstacle:
                    self.occupied.append((gx,gy))
                    if draw is not None:
                        self.draw_grid_rectangle(gx,gy,draw,"Green")

                if gy > self.grid_height:
                    self.grid_height = gy
                y = y + self.d_step
                gy = gy + 1

            if gx > self.grid_width:
                self.grid_width = gx
            x = x + self.d_step
            gx = gx + 1    

        self.grid_width += 1
        self.grid_height += 1

    def expand_obstacles(self, expansion_distance):
        #Expand all of the obstacles by the given expansion distance
        #Convert distance to a number of grid cells
        cell_exp = int(math.ceil(expansion_distance / self.meters_per_grid_cell))

        for i in range(cell_exp):
            #Expand all occupied cells by this much
            cur_occupied = []
            for x in range(self.grid_width):
                for y in range(self.grid_height):
                    if (x,y) in self.occupied:
                        #Add all neighbors, if not already occupied
                        for nx in [x-1, x, x+1]:
                            for ny in [y-1, y, y+1]:
                                if (nx,ny) not in self.occupied:
                                    if nx >= 0 and nx < self.grid_width and ny >= 0 and ny < self.grid_height:
                                        cur_occupied.append((nx,ny))
            
            #Add cells from the last expansion to our occupied list
            for p in cur_occupied:
                if p not in self.occupied:
                    self.occupied.append(p)

    def draw_grid_rectangle(self,gx,gy,draw,color):
        (x,y) = self.grid_to_image((gx,gy))
        draw.rectangle([x+1,y+1,x+self.d_step-1,y+self.d_step-1],fill=color)

    def image_to_grid(self,i):
        ix, iy = i
        gx = int(math.floor( (float(ix) / float(self.d_step))))
        gy = int(math.floor( (float(iy) / float(self.d_step))))
        return (gx,gy)

    #The pixel corresponding to each grid cell is the pixel in the upper left-hand
    #corner of the grid cell
    def grid_to_image(self,g):
        gx, gy = g
        ix = int(math.floor((float(gx) * float(self.d_step))))
        iy = int(math.floor((float(gy) * float(self.d_step))))
        return (ix,iy)

    def image_to_map(self,i):
        ix, iy = i
        mx = float(ix)*float(self.map_width)/float(self.image_width) - float(self.map_width)/2
        my = (1.0 - (float(iy)/float(self.image_height)))*float(self.map_height) - float(self.map_height)/2
        return (mx,my)

    def map_to_image(self,m):
        mx, my = m
        ix = int(math.floor(float(self.image_width) * (float(mx) + (float(self.map_width) / 2.0)) / float(self.map_width)))
        iy = int(math.floor(float(self.image_height) * (1.0 - (float(my) + (float(self.map_height) / 2.0)) / float(self.map_height))))
        return (ix,iy)

    def map_to_grid(self,m):
        mx, my = m
        i = self.map_to_image((mx,my))
        return self.image_to_grid(i)

    def grid_to_map(self, g):
        i = self.grid_to_image(g)
        return self.image_to_map(i)

    def parse_worldfile(self):
        # Parse relevant information from the world file
        # For use in various places
        inmap = False

        f = open(self.world_file)

        for line in f:
            # Get the map image (png) file
            if line.find('map') == 0:
                inmap = True
            if inmap and line.find('bitmap') != -1:
                ls = line.split('"')
                self.image_file = os.path.dirname(os.path.realpath(self.world_file)) + '/' + ls[1]
            # Get the map dimensions 
            if inmap and line.find('size') != -1:
                ls = line.split('[')
                ms = ls[1].split()
                self.map_width = float(ms[0])
                self.map_height = float(ms[1])
            if line.find('define') != -1:
                inmap = False
            # Get the robot starting location
            if line.find('turtlebot') == 0:
                ls = line.split('[')
                ms = ls[1].split()
                self.start = (int(ms[0]), int(ms[1]))
            # Get the goal location (specified as a block in the world file)
            if line.find('block') == 0:
                ls = line.split('[')
                ms = ls[1].split()
                self.goal = (int(ms[0]), int(ms[1]))

    def euclidean_distance_grid(self, a,b):
        d = math.hypot(b[0]-a[0],b[1]-a[1])
        return d

    def display_distance_LUT(self, dlut, sname):
        display_im = self.im.copy()
        draw = ImageDraw.Draw(display_im)
        max_d = 0

        for gx in range(self.grid_width-1):
            for gy in range(self.grid_height-1):
                (x,y) = self.grid_to_map((gx,gy))
                if (gx,gy) in self.occupied:
                    continue
                min_d = 40.0
                for ga in range(self.num_angles-1):
                    d = dlut.distance(x,y,self.discrete_angle_to_map_angle(ga))
                    if d < min_d:
                        min_d = d
                if min_d > max_d:
                    max_d = min_d
                rvalue = int(255 - (min_d*255/5.0))
                if rvalue < 0:
                    rvalue = 0
                self.draw_grid_rectangle(gx,gy,draw,ImageColor.getrgb('rgb(' + str(rvalue) + ',0,0)'))
                   
        #Save the image
        display_im.save(sname)
        del draw
        del display_im                   
                    

    def display_path(self, path, sname):
        # Path will be in map coordinates
        display_im = self.im.copy()
        draw = ImageDraw.Draw(display_im)

        if len(path) > 1:
            draw.line([self.map_to_image(self.start),self.grid_to_image(path[0])],fill='Green',width=3)
            for i in range(len(path)-1):
                draw.line([self.grid_to_image(path[i]), self.grid_to_image(path[i+1])],fill='Green',width=3)

        if len(path) > 1:
            draw.line([self.map_to_image(self.goal),self.grid_to_image(path[-1])],fill='Green',width=3)
        
        # Display the start (red circle) and goal (blue circle)
        gstart = self.map_to_image(self.start)
        ggoal = self.map_to_image(self.goal)
        draw.ellipse((gstart[0]-5,gstart[1]-5,gstart[0]+5,gstart[1]+5),fill='Red')
        draw.ellipse((ggoal[0]-5,ggoal[1]-5,ggoal[0]+5,ggoal[1]+5),fill='Blue')
            
        # Save the image
        display_im.save(sname)
        del draw
        del display_im

    def display_particles(self, particles, robot, laser_data, estimate, sname):
        #Create the draw object to draw with
        display_im = self.im.copy()
        draw = ImageDraw.Draw(display_im)
    
        #Plot out the particle locations on the image (in red)
        for i in range(len(particles)):
            d = 8
            color = 'Red'
            if particles[i].w == -1.0:
                color = 'Green'
            p = self.map_to_image((particles[i].x,particles[i].y))
            draw.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill=color)
            draw.line([p,(p[0] + math.cos(particles[i].theta) * d,p[1] - math.sin(particles[i].theta)*d)],fill=color,width =1)
            
        #Plot out the robot's actual location on the image (in blue)
        d = 10
        r = self.map_to_image((robot[0],robot[1]))
        draw.ellipse((r[0]-5,r[1]-5,r[0]+5,r[1]+5),fill='Blue')
        draw.line([(r[0],r[1]),(r[0]+math.cos(robot[2])*d, r[1]-math.sin(robot[2])*d)],fill='Blue',width=3)


        cur_angle = laser_data.angle_min
        
        #Plot out the current laser scan on the image (in magenta)
        for j in range(len(laser_data.ranges)):
            d2 = int(laser_data.ranges[j] * self.image_height / self.map_height)
            draw.line([(r[0],r[1]),(r[0]+math.cos(robot[2] + cur_angle)*d2, r[1]-math.sin(robot[2]+cur_angle)*d2)],fill='Magenta',width=1)
            cur_angle = cur_angle + laser_data.angle_increment
        
        #Plot out the particle filter's pose estimate on the image (in black)
        d = 8
        r = self.map_to_image((estimate[0],estimate[1]))
        draw.ellipse((r[0]-4,r[1]-4,r[0]+4,r[1]+4),fill='Black')
        draw.line([(r[0],r[1]),(r[0]+math.cos(estimate[2])*d, r[1]-math.sin(estimate[2])*d)],fill='Black',width=3)
        
        #Save the image
        display_im.save(sname)
        del draw
        del display_im

    
