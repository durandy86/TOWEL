#!/usr/bin/env python
# coding: utf-8

import numpy as np

import xarray as xr
import xarray.ufuncs as xu
import xrft
import pandas as pd

from matplotlib import pyplot as plt

import cartopy.crs as ccrs
import cartopy

plt.rc("figure", figsize=(12,10))
plt.rc("font", size=14)

### General config for plotting
map_proj = ccrs.Mercator(central_longitude=180.0, 
                         min_latitude=-20.1, 
                         max_latitude=20.1,
                         globe=None)

cmap ='jet'

    
def plotline1D(ds_FILTER, lat_Sel, lon_Sel, wave):
    da_plot= ds_FILTER[wave].sel(lat = lat_Sel, lon = lon_Sel, method = 'nearest').load()
    plt.rc("figure", figsize=(12,10))
    plt.figure()
    da_plot.plot(label = wave)
    ds_FILTER['STD_'+ wave].sel(lat = lat_Sel, lon = lon_Sel, method = 'nearest').plot(label = 'variance local', color = 'green')
    ds_FILTER['STD_'+ wave + '_N'].sel(lat = lat_Sel, lon = lon_Sel, method = 'nearest').plot(color = 'green')
    plt.legend()
    plt.ylabel('')
    plt.grid()
    
def plotline1D_mean(ds_FILTER, lat_Sel, lon_Sel, wave):
    da_plot= ds_FILTER[wave].mean(dim = 'lat').mean(dim = 'lon')
    plt.rc("figure", figsize=(12,10))
    plt.figure()
    da_plot.plot(label = wave)
    ds_FILTER['STD_'+ wave].sel(lat = lat_Sel, lon = lon_Sel, method = 'nearest').plot(label = 'variance local', color = 'green')
    ds_FILTER['STD_'+ wave + '_N'].sel(lat = lat_Sel, lon = lon_Sel, method = 'nearest').plot(color = 'green')
    plt.legend()
    plt.ylabel('')
    plt.grid()
    
def plotline1D_2V(ds_FILTER, lat_Sel, lon_Sel, wave1, wave2):
    da_plot_w1 = ds_FILTER[wave1].sel(lat = lat_Sel, lon = lon_Sel, method = 'nearest').load()
    da_plot_w2 = ds_FILTER[wave2].sel(lat = lat_Sel, lon = lon_Sel, method = 'nearest').load()

    plt.rc("figure", figsize=(12,10))
    plt.figure()
    da_plot_w1.plot(label = wave1)
    da_plot_w2.plot(label = wave2)

    ds_FILTER['STD_'+ wave1].sel(lat = lat_Sel, lon = lon_Sel).plot(label = 'variance local', color = 'green')
    ds_FILTER['STD_'+ wave1 + '_N'].sel(lat = lat_Sel, lon = lon_Sel).plot(color = 'green')
    
    ds_FILTER['STD_'+ wave2].sel(lat = lat_Sel, lon = lon_Sel).plot(label = 'variance local', color = 'purple')
    ds_FILTER['STD_'+ wave2 + '_N'].sel(lat = lat_Sel, lon = lon_Sel).plot(color = 'purple')
    plt.legend()
    plt.ylabel('')
    plt.grid()
    
def plotline1D_2V(ds_FILTER, lat_Sel, lon_Sel, wave1, wave2):
    da_plot_w1 = ds_FILTER[wave1].sel(lat = lat_Sel, lon = lon_Sel, method = 'nearest').load()
    da_plot_w2 = ds_FILTER[wave2].sel(lat = lat_Sel, lon = lon_Sel, method = 'nearest').load()

    plt.rc("figure", figsize=(12,10))
    plt.figure()
    da_plot_w1.plot(label = wave1)
    da_plot_w2.plot(label = wave2)

    ds_FILTER['STD_'+ wave1].sel(lat = lat_Sel, lon = lon_Sel, method = 'nearest').plot(label = 'variance local', color = 'green')
    ds_FILTER['STD_'+ wave1 + '_N'].sel(lat = lat_Sel, lon = lon_Sel, method = 'nearest').plot(color = 'green')
    
    ds_FILTER['STD_'+ wave2].sel(lat = lat_Sel, lon = lon_Sel, method = 'nearest').plot(label = 'variance local', color = 'purple')
    ds_FILTER['STD_'+ wave2 + '_N'].sel(lat = lat_Sel, lon = lon_Sel, method = 'nearest').plot(color = 'purple')
    plt.legend()
    plt.ylabel('')
    plt.grid()

def plot2DnbDay(ds_VAR, _da_nb_jour, _da_nb_jour_N, wave):
    plt.rc("figure", figsize=(16,4))

    #####################################################################################
    colorbar = {'label': 'OLR $W.m^{-2}$',
                'orientation': 'vertical' ,
                'extend' : 'both'}

    cmap = 'jet'
    fig, axis = plt.subplots(
        1, 1, subplot_kw={'projection': map_proj})

    ds_VAR[wave].plot.contourf(ax=axis, levels = 101, 
                        transform=ccrs.PlateCarree(),
                        cmap=cmap , 
                        cbar_kwargs=colorbar)


    axis.coastlines()
    gl = axis.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    plt.title('ecart type pour le domaine de ' + wave)
    plt.show()

    #####################################################################################
    colorbar = {'label': 'nombre',
                'orientation': 'vertical' ,
                'extend' : 'both'}

    fig, axis = plt.subplots(
        1, 1, subplot_kw={'projection': map_proj})

    _da_nb_jour.plot.contourf(ax=axis, levels = 101, 
                        transform=ccrs.PlateCarree(),
                        cmap=cmap , 
                        cbar_kwargs=colorbar)


    axis.coastlines()
    gl = axis.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    plt.title('Nombre de pas de temps supérieur a l ecart type pour ' + wave)
    plt.show()

    #####################################################################################
    fig, axis = plt.subplots(
        1, 1, subplot_kw={'projection': map_proj})

    _da_nb_jour_N.plot.contourf(ax=axis, levels = 101, 
                        transform=ccrs.PlateCarree(),
                        cmap=cmap , 
                        cbar_kwargs=colorbar)


    axis.coastlines()
    gl = axis.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    plt.title('Nombre de pas de temps inferieur a l ecart type pour ' + wave)
    plt.show()

    #####################################################################################
    fig, axis = plt.subplots(
        1, 1, subplot_kw={'projection': map_proj})

    (_da_nb_jour + _da_nb_jour_N).plot.contourf(ax=axis, levels = 101, 
                        transform=ccrs.PlateCarree(),
                        cmap=cmap , 
                        cbar_kwargs=colorbar)


    axis.coastlines()
    gl = axis.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    plt.title('Nombre de pas de temps superieur et inferieur a l ecart type pour ' + wave)
    plt.show()
    
def plot2DnbDayP(ds_FILTER, ds_VAR, _da_nb_jour, _da_nb_jour_N, wave = 'OLR_Kelvin'):
    colorbar = {'label': 'pourcentage',
                'orientation': 'vertical' ,
                'extend' : 'both'}
    
    
    _da_nb_jour_p = (_da_nb_jour / ds_FILTER.time.size)*100
    _da_nb_jour_p_N = (_da_nb_jour_N / ds_FILTER.time.size)*100
    
    fig, axis = plt.subplots(
        1, 1, subplot_kw={'projection': map_proj})

    _da_nb_jour_p.plot.contourf(ax=axis, levels = 101, 
                        transform=ccrs.PlateCarree(),
                        cmap=cmap , 
                        cbar_kwargs=colorbar)


    axis.coastlines()
    gl = axis.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    plt.title('Pourcentage de jours superieur à l ecart type pour ' + wave)  
    plt.show()
    ###################################################################################################
    fig, axis = plt.subplots(
    1, 1, subplot_kw={'projection': map_proj})

    _da_nb_jour_p_N.plot.contourf(ax=axis, levels = 101, 
                    transform=ccrs.PlateCarree(),
                    cmap=cmap , 
                    cbar_kwargs=colorbar)


    axis.coastlines()
    gl = axis.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    plt.title('Pourcentage de jours inferieur à l ecart type pour ' + wave)  
    plt.show()