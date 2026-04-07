import numpy as np
import random
from scipy.ndimage import gaussian_filter
import math

class particle:
    def __init__(self, x, y, size, terrain):
        self.pos = np.array([x + random.random() / 20, y + random.random() / 20]) # We add a little jitter to ensure that we don't land directly on a grid line
        self.velocity = np.array([0, 0])
        self.bounds = terrain.shape

        self.size = size
        self.max_size = size * 3
        self.terrain = terrain

        self.errosion_map = self.terrain * 0

        self.material = 0

    def get_normal_vec(self, pos):
        x, y = tuple(pos)

        # Get three surrounding points (on the gridline)
        a = np.array([int(np.ceil(x)), int(np.ceil(y)), self.terrain[int(np.ceil(x)), int(np.ceil(y))] ])
        b = np.array([int(np.floor(x)), int(np.ceil(y)), self.terrain[int(np.floor(x)), int(np.ceil(y))]])
        c = np.array([int(np.floor(x)), int(np.floor(y)), self.terrain[int(np.floor(x)), int(np.floor(y))]])

        
        # Choose two vectors bordering our cell, cross them to get the normal vector
        norm = np.cross(b-a, c-a)

        # Normalize and return
        normal_norm = (norm @ norm.T) ** 0.5

        if normal_norm > 0:
            return norm / normal_norm
        return norm

    def normalize(self, point):
        return np.array([ point[0] / self.bounds[0],  point[1] / self.bounds[1] ])

    def interp_z(self, x, y):
        # Get the normal plane to where we are
        a, b, c = tuple(self.get_normal_vec( (x, y) ))

        # Grab an adjacent point on the grid, interpret it as a point on the plane
        px = int(x)
        py = int(y)
        pz = self.terrain[px, py]

        # Calculate our plane intercept
        d = a * px + b * py + c * pz

        # Now estimate adjacent z along the plane
        return (-1 * (a*x + b*y) + d) / c

    # A simple step in our physics simulation
    def step(self, stepsize=0.5, friction = 0.3):

        # Find the direction of steepest slope
        norm =  self.get_normal_vec(self.pos)
        down = np.array([0, 0, -1])

        # Project downwards vector onto our plane
        slope = down - (down @ norm) * norm

        # Normalize it
        slope_norm = (slope @ slope.T) ** 0.5
        if slope_norm > 0:
            slope = slope / slope_norm 

        # sin and cos of the slope (incline slope problem)
        sin_theta = -1 * slope[2]
        cos_theta = ( slope[0] ** 2 + slope[1] ** 2 ) ** 0.5

        # Just a Physics I incline slope problem (metric system)
        a = max(9.8 * (sin_theta - cos_theta * friction), 0)
        a_flattened = cos_theta * a

        velocity_vector = (a * slope[:2] ) * stepsize

        # Note our current position, then iterate it
        old_pos = self.pos
        next_pos = self.pos + (self.velocity + velocity_vector) * stepsize

        if not self.in_bounds(next_pos):
            return

        # Now let's do a potential energy check (friction ignored)
        old_z = self.interp_z(old_pos[0], old_pos[1])
        next_z = self.interp_z(next_pos[0], next_pos[1])

        delta_z = next_z - old_z

        v_2 = ( (self.velocity + velocity_vector) @ (self.velocity + velocity_vector).T )

        if (2 * 9.8 * delta_z) > v_2:
            # Hit an uphill and we don't have enough energy to go where we're trying to go.
            """
            Find the velocity (momentary acceleration) vector at our target location, average it out with our current one
            """
            # Find the direction of steepest slope at our target location
            next_norm =  self.get_normal_vec(next_pos)

            # Project downwards vector onto our plane
            next_slope = down - (down @ norm) * norm

            # Normalize it
            next_slope_norm = (next_slope @ next_slope.T) ** 0.5
            if next_slope_norm > 0:
                next_slope = next_slope / next_slope_norm 

            # sin and cos of the slope (incline slope problem)
            sin_theta = -1 * next_slope[2]
            cos_theta = ( next_slope[0] ** 2 + next_slope[1] ** 2 ) ** 0.5

            # Just a Physics I incline slope problem (metric system)
            a = max(9.8 * (sin_theta - cos_theta * friction), 0)
            a_flattened = cos_theta * a

            next_velocity_vector = (a * next_slope[:2] ) * stepsize * 2 # This 2 makes it a weighted sum (looks more natural)
            avg_vector = (next_velocity_vector + velocity_vector) / 2

            # Update our velocity and position with this averaged out vector
            self.velocity = self.velocity + avg_vector
            self.pos = self.pos + self.velocity * stepsize

        else:
            # Everything's valid, just update our velocity and position
            self.velocity = self.velocity + velocity_vector
            self.pos = next_pos



    def in_bounds(self, point):
        return not (point[0] <= 1 or point[0] >= self.bounds[0] - 2 or point[1] <= 1 or point[1] >= self.bounds[1]-2)

    
    # Given a non-integer point, let's get all of the adjacent nodes in our height map (for alterations)
    def get_adjacent_squares(self, x, y):
        zeros = self.terrain * 0

        zeros[int(np.ceil(x)), int(np.ceil(y))] = 0.25
        zeros[int(np.ceil(x)), int(np.floor(y))] = 0.25
        zeros[int(np.floor(x)), int(np.ceil(y))] = 0.25
        zeros[int(np.floor(x)), int(np.floor(y))] = 0.25

        return zeros

    # Run our simulation for a given number of steps, calculating erosion
    def simulate(self, times = 10, stepsize=0.5, friction=0.5, dissolution_rate = 0.1):
        pos_vec = []
        
        for i in range(times):
            if not self.in_bounds(self.pos):
                return pos_vec, self.errosion_map

            # Calculate and log current position
            pre_x, pre_y = self.pos
            pre_z = self.interp_z(pre_x, pre_y)
            pos_1 = ( pre_x, pre_y , pre_z)

            pos_vec.append( pos_1 )

            # Move our position
            self.step(stepsize, friction)
            continue

            # If we went out of bounds, then return
            if not self.in_bounds(self.pos):
                return pos_vec, self.errosion_map

            # Find out position after moving
            post_x, post_y = self.pos
            post_z = self.interp_z(post_x, post_y)

            # Calculate carrying capacity/erosion
            xslope = self.terrain[round(self.pos[0]-1), round(self.pos[1])] - self.terrain[round(self.pos[0]+1), round(self.pos[1])] 
            yslope = self.terrain[round(self.pos[0]), round(self.pos[1]-1)] - self.terrain[round(self.pos[0]), round(self.pos[1]+1)]  

            gradient = np.array([xslope, yslope])
            capacity = self.size * (self.velocity @ self.velocity.T) ** 0.5 #* (gradient @ gradient.T) ** 0.5  slope * speed * size
            capacity = min(capacity, self.max_size)

            if self.material < capacity:
                amount_erroded = min( (capacity - self.material) * dissolution_rate * stepsize, abs(pre_z - post_z) )
            else:
                amount_erroded = (capacity - self.material) * stepsize # in this case, we have more sediment than the droplet can hold, start shedding weight
                        
            self.material += amount_erroded
            self.errosion_map[round(pre_x), round(pre_y)] = self.errosion_map[round(pre_x), round(pre_y)] - amount_erroded

            self.size = self.size * 0.98
            if self.size <= 0.01:
                return pos_vec, self.erosion_map

        # If we settle at the end of the simulation, then we need to deposit all of the material
        x, y = self.pos
        #self.errosion_map = self.errosion_map + self.get_adjacent_squares(x, y) * self.material
        self.material = 0

        return pos_vec, self.errosion_map
