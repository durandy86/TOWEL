#!/usr/bin/env python
# coding: utf-8

import numpy as np

import xarray as xr
import xarray.ufuncs as xu
import xrft
import pandas as pd


def addNegative(ds_FILTER, ds_std, wave):
    # input : 
    # ds_FILTER = dataset of the anomaly filtered
    # ds_std* = dataset of the standard deviation for the anomaly filtered
    # wave = name
    # output : 
    # dataset with new variables 'STD_' + wave corresponding to the standard deviation
    ### Add negatif standard deviation to the DATAs

    # Local
    ds_FILTER['STD_' + wave ] = ds_FILTER[wave] / ds_FILTER[wave]
    ds_FILTER['STD_' + wave ] = (ds_FILTER['STD_'+ wave] * ds_std[wave].values)
    ds_FILTER['STD_' + wave + '_N']  = -1 * ds_FILTER['STD_'+ wave] 
    return ds_FILTER

###########################################################################
def keepData(ds_FILTER, coeff, wave_REF, wave_CROSS):
    ds_below = ds_FILTER.copy()
    ds_above = ds_FILTER.copy()

    ds_below[wave_REF] = xr.where(ds_FILTER[wave_REF] <= (ds_FILTER['STD_' + wave_REF +'_N'] * coeff),
                       ds_FILTER[wave_REF], 0)
    ds_above[wave_REF] = xr.where(ds_FILTER[wave_REF] >= (ds_FILTER['STD_' + wave_REF] * coeff),
                       ds_FILTER[wave_REF], 0)
    
    ### Create new variable
    ds_FILTER[wave_REF + '_sum_TS'] = ds_below[wave_REF] + ds_above[wave_REF]
    ds_FILTER[wave_REF + '_below_TS'] = ds_below[wave_REF]
    ds_FILTER[wave_REF + '_above_TS'] = ds_above[wave_REF]

    ds_FILTER[wave_CROSS + '_FILTERED_by_' + wave_REF +'_TS'] = xr.where((ds_FILTER[wave_REF] >= (ds_FILTER['STD_' + wave_REF] * coeff)) | (ds_FILTER[wave_REF] <= (ds_FILTER['STD_' + wave_REF + '_N'] * coeff)) == True,
                            ds_FILTER[wave_CROSS], 0)
    return ds_FILTER


###########################################################################
###########################################################################
###########################################################################
def openDATA(year, lat_min_box, lat_max_box, lon_min_box, lon_max_box):
    #########
    ### v ###
    #########
    indir_data_RAW = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/RAW_ANOMALY/v/'
    indir_data_FILTERED = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/FILTERED_ANOMALY/v/'
    var_file = '*'

    ### Open DATA
    ds_RAW = xr.open_mfdataset(indir_data_RAW+'*'+var_file+'*'+str(year)+'.nc', chunks = {'time' : 1, 'level' : 1}, 
                               parallel=True, combine = 'nested', concat_dim = 'level')
    ds_RAW = ds_RAW.rename({'latitude':'lat', 'longitude':'lon'})
    ds_RAW = ds_RAW.sel(level = [850,200])
    ds_RAW = ds_RAW.assign_coords(lon = (((ds_RAW.lon + 180) % 360) - 180)).sortby('lon')

    ds_FILTER = xr.open_mfdataset(indir_data_FILTERED + '*' + str(year) + '.nc', chunks = {'time' : 1}, parallel=True)
    ds_FILTER = ds_FILTER.assign_coords(lon = (((ds_FILTER.lon + 180) % 360) - 180)).sortby('lon')

    ds_RAW = ds_RAW.isel(lat = slice(2,None,4), lon = slice(2,None,4))
    ds_FILTER = ds_FILTER.isel(lat = slice(2,None,4), lon = slice(2,None,4))

    ds_RAW = ds_RAW.sel(lat = slice(lat_max_box, lat_min_box), lon = slice(lon_min_box, lon_max_box))
    ds_FILTER = ds_FILTER.sel(lat = slice(lat_max_box, lat_min_box), lon = slice(lon_min_box, lon_max_box))

    ds_RAW_v = ds_RAW.reindex(lat=list(reversed(ds_RAW.lat)))
    ds_FILTER_v = ds_FILTER.reindex(lat=list(reversed(ds_FILTER.lat)))
    
    #########
    ### u ###
    #########
    indir_data_RAW = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/RAW_ANOMALY/u/'
    indir_data_FILTERED = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/FILTERED_ANOMALY/u/'
    var_file = '*'

    ### Open DATA
    ds_RAW = xr.open_mfdataset(indir_data_RAW+'*'+var_file+'*'+str(year)+'.nc', chunks = {'time' : 1, 'level' : 1}, parallel=True,
                              combine = 'nested', concat_dim = 'level')
    ds_RAW = ds_RAW.rename({'latitude':'lat', 'longitude':'lon'})
    ds_RAW = ds_RAW.sel(level = [850,200])
    ds_RAW = ds_RAW.assign_coords(lon = (((ds_RAW.lon + 180) % 360) - 180)).sortby('lon')

    ds_FILTER = xr.open_mfdataset(indir_data_FILTERED + '*' + str(year) + '.nc', chunks = {'time' : 1}, parallel=True)
    ds_FILTER = ds_FILTER.assign_coords(lon = (((ds_FILTER.lon + 180) % 360) - 180)).sortby('lon')


    ds_RAW = ds_RAW.isel(lat = slice(2,None,4), lon = slice(2,None,4))
    ds_FILTER = ds_FILTER.isel(lat = slice(2,None,4), lon = slice(2,None,4))

    ds_RAW = ds_RAW.sel(lat = slice(lat_max_box, lat_min_box), lon = slice(lon_min_box, lon_max_box))
    ds_FILTER = ds_FILTER.sel(lat = slice(lat_max_box, lat_min_box), lon = slice(lon_min_box, lon_max_box))
    
    ds_RAW_u = ds_RAW.reindex(lat=list(reversed(ds_RAW.lat)))
    ds_FILTER_u = ds_FILTER.reindex(lat=list(reversed(ds_FILTER.lat)))
    
    ############
    ### TCWV ###
    ############
    indir_data_RAW = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/RAW_ANOMALY/TCWV/'
    indir_data_FILTERED = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/FILTERED_ANOMALY/TCWV/'
    indir_data_VAR = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/ANALYSIS/VARIANCE/ANOMALY_FILTERED/'
    indir_data_VAR_TOT = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/ANALYSIS/VARIANCE/RAW_ANOMALY/'
    var_file = '*'

    ### Open DATA
    ds_RAW = xr.open_mfdataset(indir_data_RAW+'*'+var_file+'*'+str(year)+'.nc', chunks = {'time' : 1}, parallel=True)
    ds_RAW = ds_RAW.rename({'latitude':'lat', 'longitude':'lon'})
    ds_RAW = ds_RAW.assign_coords(lon = (((ds_RAW.lon + 180) % 360) - 180)).sortby('lon')


    ds_FILTER = xr.open_mfdataset(indir_data_FILTERED + '*' + str(year) + '.nc', chunks = {'time' : 1}, parallel=True)
    ds_FILTER = ds_FILTER.assign_coords(lon = (((ds_FILTER.lon + 180) % 360) - 180)).sortby('lon')


    ds_VAR = xr.open_mfdataset(indir_data_VAR + 'TCWV_YEAR.nc', chunks = {'time' : 1}, parallel=True)
    ds_VAR_TOT = xr.open_mfdataset(indir_data_VAR_TOT + 'TCWV_YEAR.nc', chunks = {'time' : 1}, parallel=True)
    ds_VAR_TOT = ds_VAR_TOT.rename({'latitude':'lat', 'longitude':'lon'})

    ds_RAW = ds_RAW.isel(lat = slice(2,None,4), lon = slice(2,None,4))
    ds_FILTER = ds_FILTER.isel(lat = slice(2,None,4), lon = slice(2,None,4))
    ds_VAR = ds_VAR.isel(lat = slice(2,None,4), lon = slice(2,None,4))
    ds_VAR_TOT = ds_VAR_TOT.isel(lat = slice(2,None,4), lon = slice(2,None,4))

    ds_RAW = ds_RAW.sel(lat = slice(lat_max_box, lat_min_box), lon = slice(lon_min_box, lon_max_box))
    ds_FILTER = ds_FILTER.sel(lat = slice(lat_max_box, lat_min_box), lon = slice(lon_min_box, lon_max_box))
    ds_VAR = ds_VAR.sel(lat = slice(30, -30))
    ds_VAR_TOT = ds_VAR_TOT.sel(lat = slice(30, -30))

    ds_RAW_TCWV = ds_RAW.reindex(lat=list(reversed(ds_RAW.lat)))
    ds_FILTER_TCWV = ds_FILTER.reindex(lat=list(reversed(ds_FILTER.lat)))
    ds_VAR_TCWV = ds_VAR.reindex(lat=list(reversed(ds_VAR.lat)))
    ds_VAR_TOT_TCWV = ds_VAR_TOT.reindex(lat=list(reversed(ds_VAR_TOT.lat)))

    ###########
    ### OLR ###
    ###########
    
    indir_data_RAW = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/RAW_ANOMALY/OLR/'
    indir_data_FILTERED = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/FILTERED_ANOMALY/OLR/'
    indir_data_VAR = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/ANALYSIS/VARIANCE/ANOMALY_FILTERED/'
    indir_data_VAR_TOT = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/ANALYSIS/VARIANCE/RAW_ANOMALY/'

    var_file = 'anom_OLR_brut_ERA5_3H'
    
    ### Open DATA
    ds_RAW_OLR = xr.open_mfdataset(indir_data_RAW+'*'+var_file+'*'+str(year)+'.nc', chunks = {'time' : 1}, parallel=True)
    ds_FILTER_OLR = xr.open_mfdataset(indir_data_FILTERED + '*' + str(year) + '.nc', chunks = {'time' : 1}, parallel=True)
    ds_VAR_OLR = xr.open_mfdataset(indir_data_VAR + 'OLR_YEAR.nc', chunks = {'time' : 1}, parallel=True)
    ds_VAR_TOT = xr.open_mfdataset(indir_data_VAR_TOT + 'OLR_YEAR.nc', chunks = {'time' : 1}, parallel=True)
    
    ### Reasign coords
    ds_VAR_TCWV = ds_VAR_TCWV.assign_coords(lon = (((ds_VAR.lon + 180) % 360) - 180)).sortby('lon')
    ds_FILTER_OLR = ds_FILTER_OLR.assign_coords(lon = (((ds_FILTER_OLR.lon + 180) % 360) - 180)).sortby('lon')
    ds_VAR_OLR = ds_VAR_OLR.assign_coords(lon = (((ds_VAR.lon + 180) % 360) - 180)).sortby('lon')
    ds_RAW_OLR = ds_RAW_OLR.assign_coords(lon = (((ds_RAW_OLR.lon + 180) % 360) - 180)).sortby('lon')
    
    ds_FILTER_OLR = ds_FILTER_OLR.sel(lat = slice(lat_min_box, lat_max_box), lon = slice(lon_min_box, lon_max_box))
    ds_RAW_OLR = ds_RAW_OLR.sel(lat = slice(lat_min_box, lat_max_box), lon = slice(lon_min_box, lon_max_box))
    
    del ds_RAW, ds_VAR, ds_FILTER
    
    ds_VAR = xr.merge([ds_VAR_TCWV, ds_VAR_OLR])
    ds_VAR = ds_VAR.sel(lat = slice(lat_min_box, lat_max_box), lon = slice(lon_min_box, lon_max_box))
    ds = xr.merge([ds_FILTER_u, ds_FILTER_v, ds_FILTER_TCWV, ds_FILTER_OLR, ds_RAW_u, ds_RAW_v, ds_RAW_TCWV, ds_RAW_OLR])
    ds = ds.sel(lat = slice(lat_min_box, lat_max_box), lon = slice(lon_min_box, lon_max_box))
    
    return ds_VAR, ds