import numpy as np
from tqdm import tqdm
from particle import particle
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from noise import pnoise2


def int_to_gradient(value, min_val=0, max_val=10):
    value = max(min_val, min(value, max_val))
    t = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.0
    r1, g1, b1 = 0, 0, 139   # #00008B
    r2, g2, b2 = 255, 0, 0  # #FF0000
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)

    return f"#{r:02X}{g:02X}{b:02X}"

def convert_to_surface(terrain_map):
    WIDTH, DEPTH = terrain_map.shape
    x = np.linspace(1, WIDTH, WIDTH)
    y = np.linspace(1, DEPTH, DEPTH)
    X, Y = np.meshgrid(x, y)
    Z = terrain_map

    return Y, X, Z

def display_terrain(terrain_map, ax, wireframe=False):
    X, Y, Z = convert_to_surface(terrain_map)

    if wireframe:
        ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, color='black')  # wireframe only
    else:
        surface = ax.plot_surface(
            X, Y, Z,
            rstride=1,
            cstride=1,
            cmap='terrain',   # good for heightmaps
            linewidth=0,
            antialiased=True
        )
        fig.colorbar(surface, shrink=0.5, aspect=10)

    ax.set_title("Perlin Noise Heightmap")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")


def display_points(pointset, ax):
    for i, points in enumerate(pointset):
        point_x = [point[0] for point in points]
        point_y = [point[1] for point in points]
        point_z = [point[2] for point in points]

        ax.plot(
            point_x, point_y, point_z,
            linestyle='',
            marker='o',
            markersize=10,                 # bigger = more visible
            markerfacecolor=int_to_gradient(i),     # bright blue (DeepSkyBlue)
            markeredgewidth=1.5
        )

def display_particle(gradient, slope_vec, velocity, pos, ax):
    ax.plot(pos[0], pos[1], pos[2],
            linestyle='',
            marker='o',
            markersize=10,                 # bigger = more visible
            markerfacecolor='#00BFFF',     # bright blue (DeepSkyBlue)
            markeredgecolor='white',       # high contrast outline
            markeredgewidth=1.5)

    ax.quiver(
        pos[0], pos[1], pos[2],   # starting points of arrows
        velocity[0], velocity[1], 0,   # vector components
        arrow_length_ratio = 0.01,
        length=10,  # scale of arrows
        color='red'
    )

    ax.quiver(
        pos[0], pos[1], pos[2],   # starting points of arrows
        slope_vec[0], slope_vec[1], slope_vec[2],   # vector components
        arrow_length_ratio = 0.01,
        length=10,  # scale of arrows
        color='blue'
    )



def get_perlin(WIDTH = 100, DEPTH = 100, seed=random.randint(0,1000)):
    scale = min(WIDTH, DEPTH)       # Larger = smoother noise
    octaves = 5         # Layers of detail
    persistence = 0.4   # Amplitude reduction per octave
    lacunarity = 2    # Frequency increase per octave

    noise_map = np.zeros((WIDTH, DEPTH))

    for y in range(DEPTH):
        for x in range(WIDTH):
            noise_map[y][x] = pnoise2(
                x / scale,
                y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=seed) * (WIDTH + DEPTH) / 2

    max_height = max([max(i) for i in noise_map])
    min_height = min([min(i) for i in noise_map])

    height_scale = (WIDTH + DEPTH) /2

    return (noise_map - min_height) * ( height_scale/ (max_height - min_height) )

if __name__ == "__main__":

    perlin_map = get_perlin(100, 100, 3)
    #perlin_map = np.array([np.linspace((i -50)**2,(i -50)**2,100) for i in range(100)])

    raindrops = []

    errosion = perlin_map * 0

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')  # left
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')  # left
    display_terrain(perlin_map, ax1)


    for i in tqdm(range(200)): 
        raindrop = particle(random.randint(0, 99), random.randint(0,99), 1, perlin_map)
         
        points, errosion_map = raindrop.simulate(100, 0.5, 0.3, 1)
        raindrops.append( points )

        errosion = errosion + errosion_map
        perlin_map = perlin_map + errosion
    

    display_terrain(perlin_map, ax2)
    display_points(raindrops, ax2)
    plt.show()

