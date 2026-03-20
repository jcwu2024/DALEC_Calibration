# Description: This script contains functions to convert between latitude/longitude and land index.
import math

def latlon2land(lat, lon, res=0.5):
    lat_grid = (math.floor(lat / res) + res) * res
    lon_grid = (math.floor(lon / res) + res) * res

    ncols = int(360 / res)
    row_index = int((90 - lat_grid) / res - res)
    col_index = int((lon_grid + 180) / res - res)
    land_value = row_index * ncols + col_index + 1
    return land_value

def land2latlon(land, res=0.5):
    ncols = int(360 / res)
    row_index = (land - 1) // ncols
    col_index = (land - 1) % ncols
    lat = 90 - (row_index + res) * res
    lon = (col_index + res) * res - 180
    return lat, lon