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

    def get_adjacent_points(self, pos):
        x, y = tuple(pos)
        # Get three surrounding points (on the gridline)
        a = np.array([int(np.floor(x)+1), int(np.floor(y)+1), self.terrain[int(np.floor(x)+1), int(np.floor(y)+1)] ])
        b = np.array([int(np.floor(x)), int(np.floor(y)+1), self.terrain[int(np.floor(x)), int(np.floor(y)+1)]])
        c = np.array([int(np.floor(x)), int(np.floor(y)), self.terrain[int(np.floor(x)), int(np.floor(y))]])
        d = np.array([int(np.floor(x)+1), int(np.floor(y)), self.terrain[int(np.floor(x)+1), int(np.floor(y))]])

        x_margin = x - np.floor(x)
        y_margin = y - np.floor(y)

        if (1 - x_margin) < y_margin:
            return (a, b, d)
        return(a, c, d)
        

    def get_normal_vec(self, pos):

        a, b, c = self.get_adjacent_points(pos)
        
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
        """
        Find the direction of our steepset slope
        """
        norm =  self.get_normal_vec(self.pos)
        down = np.array([0, 0, -1])

        # Project downwards vector onto our plane
        slope = down - (down @ norm) * norm

        # Normalize it
        slope_norm = (slope @ slope.T) ** 0.5
        if slope_norm > 0:
            slope = slope / slope_norm 

        """
        Calculate momentary acceleration given slope & Friction
        """
        sin_theta = -1 * slope[2]
        cos_theta = ( slope[0] ** 2 + slope[1] ** 2 ) ** 0.5

        # Just a Physics I incline slope problem (metric system)
        a = max(9.8 * (sin_theta - cos_theta * friction), 0)
        a_flattened = cos_theta * a

        velocity_vector = (a * slope[:2] ) * stepsize

        """
        Iterate position
        """
        old_pos = self.pos
        next_pos = self.pos + (self.velocity + velocity_vector) * stepsize

        if not self.in_bounds(next_pos):
            return

        """
        This is a potential energy check
        """
        old_z = self.interp_z(old_pos[0], old_pos[1])
        next_z = self.interp_z(next_pos[0], next_pos[1])

        delta_z = next_z - old_z

        v_2 = ( (self.velocity + velocity_vector) @ (self.velocity + velocity_vector).T )

        if (2 * 9.8 * delta_z) > v_2:
            # Hit an uphill and we don't have enough energy to go where we're trying to go.
            """
            Find the velocity (momentary acceleration) vector at our target location, average it out with our current one (weighted avg)
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

    def errode(self, pos, amount_erroded):
        a, b, c = self.get_adjacent_points(pos)

        base = min(( a[2], b[2], c[2]))
        top = max(( a[2], b[2], c[2]))

        # Based on how sloped the current cell is, we want to limit erosion
        if abs(amount_erroded) > (top - base) *4:
            amount_erroded = (top - base) * 4 * (amount_erroded / abs(amount_erroded)) # Maintain the sign of our erosion (for depositing material)

        if amount_erroded > 0:
            reduced_max = top - amount_erroded / 4
            for q in [a,b,c]:
                reduced_q = (q[2] - base) * (reduced_max - base) / (top - base) + base
                self.errosion_map[ int(q[0]), int(q[1])] -= (q[2] - reduced_q)

        if amount_erroded < 0:
            increased_min = base - amount_erroded / 4

            for q in [a,b,c]:
                increased_q = top - (top - q[2]) * (top - increased_min) / (top - base)
                self.errosion_map[ int(q[0]), int(q[1])] -= (q[2] - increased_q)

        return  amount_erroded

    # Run our simulation for a given number of steps, calculating erosion
    def simulate(self, times = 10, stepsize=0.5, friction=0.5, dissolution_rate = 0.1):
        pos_vec = []
        
        for i in range(times):
            if not self.in_bounds(self.pos):
                return pos_vec, self.errosion_map

            # Calculate and log current position
            pre_x, pre_y = self.pos
            pre_z = self.interp_z(pre_x, pre_y)

            pos_vec.append( ( pre_x, pre_y , pre_z) )

            # Move our position
            self.step(stepsize, friction)

            # If we went out of bounds, then return
            if not self.in_bounds(self.pos):
                return pos_vec, self.errosion_map

            # Momentary capacity is affected by velocity
            capacity = self.size * (self.velocity @ self.velocity.T) ** 0.5 
            capacity = min(capacity, self.max_size)

            if self.material < capacity:
                amount_erroded = (capacity - self.material) * dissolution_rate * stepsize
            else:
                amount_erroded = (capacity - self.material) * stepsize # in this case, we have more sediment than the droplet can hold, start shedding weight

            # Do our actual errosion
            amount_erroded = self.errode(self.pos, amount_erroded)

            self.material += amount_erroded

            # Decrease the size by a little bit (evaporation)
            self.size = self.size * 0.9

            if self.size <= 0.01:
                break

        # If we settle at the end of the simulation, then we need to deposit all of the material
        self.errode(self.pos, -1 * self.material)
        self.material = 0

        return pos_vec, self.errosion_map
