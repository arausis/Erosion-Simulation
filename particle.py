import numpy as np
from scipy.ndimage import gaussian_filter
import math

class particle:
    def __init__(self, x, y, size, terrain):
        self.pos = np.array([x + 0.01, y + 0.01]) # We add a little jitter to ensure that we don't land directly on a grid line
        self.velocity = np.array([0, 0])
        self.bounds = terrain.shape

        self.size = size
        self.max_size = size * 3
        self.terrain = terrain

        self.errosion_map = self.terrain * 0

        self.material = 0

    # Use space of current x,y and slope to approximate the z at a non-integer space. Note, should not be used far away from current position
    def get_normal_vec(self, pos):

        x, y = tuple(pos)

        # Get all of the surrounding points
        a = np.array([int(np.ceil(x)), int(np.ceil(y)), self.terrain[int(np.ceil(x)), int(np.ceil(y))] ])
        b = np.array([int(np.floor(x)), int(np.ceil(y)), self.terrain[int(np.floor(x)), int(np.ceil(y))]])
        c = np.array([int(np.floor(x)), int(np.floor(y)), self.terrain[int(np.floor(x)), int(np.floor(y))]])

        norm = np.cross(b-a, c-a)

        return norm / (norm @ norm.T) ** 0.5


    def step(self, stepsize=0.5, friction = 0.3):
        xslope = self.terrain[round(self.pos[0]-1), round(self.pos[1])] - self.terrain[round(self.pos[0]+1), round(self.pos[1])] 
        yslope = self.terrain[round(self.pos[0]), round(self.pos[1]-1)] - self.terrain[round(self.pos[0]), round(self.pos[1]+1)]  

        #Alright, that's the direction of steepest slope
        gradient = np.array([xslope, yslope])
        gnorm = (gradient @ gradient.T)**0.5
        if gnorm > 0:
            gradient = gradient / gnorm
        n_vec = self.get_normal_vec(self.pos)


        # Now get a simple 2d incline slope in that direction, project and normalize
        slope_vec = np.array([xslope, yslope, 0])
        slope_vec = slope_vec - (slope_vec @ n_vec.T)
        slope_norm = (slope_vec @ slope_vec.T) ** 0.5
        if slope_norm > 0:
            slope_vec =  slope_vec /  slope_norm

        # Normalize our slope vector and then run some basic calculations
        sin_theta = -1 * slope_vec[2]
        cos_theta = (slope_vec[0]** 2 + slope_vec[1]**2) ** 0.5


        # Physics I incline slope problem fr (guess we're using metric btw)
        a = max(9.8 * (sin_theta - cos_theta * friction), 0)
        a_flattened = cos_theta * a

        self.velocity = self.velocity + (a_flattened * gradient ) * stepsize

        #Given the slope of our current pannel,
        old_pos = self.pos
        self.pos = self.pos + self.velocity * stepsize

        return gradient, slope_vec, self.velocity, old_pos

    def in_bounds(self, point):
        return not (point[0] <= 1 or point[0] >= self.bounds[0] - 2 or point[1] <= 1 or point[1] >= self.bounds[1]-2)

    def normalize(self, point):
        return np.array([ point[0] / self.bounds[0],  point[1] / self.bounds[1] ])

    def interp_x(self, x, y):
        # Get the normal plane to where we are
        a, b, c = tuple(self.get_normal_vec( (x, y) ))

        # Grab an adjacent point on the grid, interpret it as a point on the plane
        px = int(x)
        py = int(x)
        pz = self.terrain[px, py]

        # Calculate our plane intercept
        d = a * px + b * py + c * pz

        # Now estimate adjacent z along the plane
        return (-1 * (a*x + b*y) + d) / c
    
    # Given a non-integer point, let's get all of the adjacent nodes in our height map (for alterations)
    def get_adjacent_squares(self, x, y):
        zeros = self.terrain * 0

        zeros[int(np.ceil(x)), int(np.ceil(y))] = 0.25
        zeros[int(np.ceil(x)), int(np.floor(y))] = 0.25
        zeros[int(np.floor(x)), int(np.ceil(y))] = 0.25
        zeros[int(np.floor(x)), int(np.floor(y))] = 0.25

        return zeros

    def simulate(self, times = 10, stepsize=0.5, friction=0.5, dissolution_rate = 0.1):
        pos_vec = []
        
        for i in range(times):
            if not self.in_bounds(self.pos):
                return pos_vec, self.errosion_map

            # Calculate and log current position
            pre_x, pre_y = self.pos
            pre_z = self.interp_x(pre_x, pre_y)
            pos_1 = ( pre_x, pre_y , pre_z)

            pos_vec.append( pos_1 )

            # Move our position
            self.step(stepsize, friction)

            # If we went out of bounds, then return
            if not self.in_bounds(self.pos):
                return pos_vec, self.errosion_map

            # Find out position after moving
            post_x, post_y = self.pos
            post_z = self.interp_x(post_x, post_y)

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
        self.errosion_map = self.errosion_map + self.get_adjacent_squares(x, y) * self.material
        self.material = 0

        return pos_vec, self.errosion_map
