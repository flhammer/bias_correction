import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


class FigureManager:
    def __init__(self):
        self.figures = {}

    def create_new(self):
        now = datetime.now()
        (fig, ax) = plt.subplots()
        self.figures[now.isoformat()] = fig
        return (fig, ax)

    def close_all(self):
        [plt.close(fig) for fig in self.figures.values()]
        self.figures = {}


def add_boundaries_to_plot(ax, minx=None, maxx=None, miny=None, maxy=None):
    ax.plot([minx, maxx], [maxy, maxy], color="black", linewidth=2)
    ax.plot([maxx, maxx], [miny, maxy], color="black", linewidth=2)
    ax.plot([minx, maxx], [miny, miny], color="black", linewidth=2)
    ax.plot([minx, minx], [miny, maxy], color="black", linewidth=2)


def get_subdomain(x0, y0):
    deltax = 1000
    deltay = 1000
    return dict(
        minx=x0,
        maxx=x0+deltax,
        miny=y0,
        maxy=y0+deltay
    )


def compute_devine_wind(unet, angle):
    # angle: angle en degrés "météorologique"
    unet_for_angle = unet.isel(angle=angle)
    alpha_wind = np.mod(3*np.pi/2 - np.deg2rad(angle) + unet_for_angle.alpha, 2*np.pi)
    windspeed = 3*unet_for_angle.acceleration
    x, y = np.meshgrid(unet_for_angle.x.values, unet_for_angle.y.values)
    u = np.cos(alpha_wind) * windspeed
    v = np.sin(alpha_wind) * windspeed
    return x, y, u, v
