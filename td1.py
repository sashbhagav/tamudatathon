"""
TAMU DATATHON 2019
Author: John Gutierrez
File Purpose:
    Establish map of US using plt and gp
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

FIG_X, FIG_Y = 12, 7
states_of_interest = ["Texas"]


def state_plotter(states, axis, us_map=True):
    """Takes a list of states and plots them with highlight. If us_map is True(default), will print background states.
    *adjusts the frame to scale with AK, HI."""

    # has background of all states
    if us_map:
        # need to adjust the scale if outlier states are present
        if 'HI' in states:
            usa_gpd[0:50].plot(ax=axis, alpha=0.3)
        elif 'AK' in states:
            usa_gpd[0:51].plot(ax=axis, alpha=0.3)
        elif 'HI' and 'AK' in states:
            usa_gpd[1:50].plot(ax=axis, alpha=0.3)
        else:
            usa_gpd[1:50].plot(ax=axis, alpha=0.3)

        for s in states:
            usa_gpd[usa_gpd.STATE_ABBR == f'{s}'].plot(ax=axis, edgecolor='y', linewidth=2)

    # no background outline of all states
    elif not us_map:
        for s in states:
            usa_gpd[usa_gpd.STATE_ABBR == f'{s}'].plot(ax=axis, edgecolor='y', linewidth=2)


def zip_to_lonlat(zip_code):
    """Gets the longitude and latitude for a given zip code"""
    return zip_code, zip_code


# states from shp downloaded file
usa_gpd = gpd.read_file("states_21basic/states.shp")

# read in taco data
taco_burrito_dat = gpd.read_file("just tacos and burritos.csv")
geometry = [Point(xy) for xy in zip(taco_burrito_dat['longitude'], taco_burrito_dat['latitude'])]
taco_burrito_gpd = gpd.GeoDataFrame(taco_burrito_dat, crs=usa_gpd.crs, geometry=geometry)


# read in taco data
taco_dat = gpd.read_file("just tacos and burritos.csv")


# plotting
fig, ax = plt.subplots(figsize=(FIG_X, FIG_Y))
ax.set_aspect('equal')
state_plotter(states_of_interest, ax, us_map=True)
taco_burrito_gpd.plot(ax=ax, color="g", markersize=0.5)


plt.show()
