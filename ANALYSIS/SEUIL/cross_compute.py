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
    ds_FILTER['STD_' + wave ] = ds_FILTER[wave]*0. + 1.
    ds_FILTER['STD_' + wave ] = (ds_FILTER['STD_'+ wave] * ds_std[wave])
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
def openDATA(year, season):
    ############
    ### TCWV ###
    ############
    indir_data_RAW = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/RAW_ANOMALY/TCWV/'
    indir_data_FILTERED = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/FILTERED_ANOMALY/TCWV/'
    indir_data_VAR = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/ANALYSIS/VARIANCE/ANOMALY_FILTERED/'
    indir_data_VAR_TOT = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/ANALYSIS/VARIANCE/RAW_ANOMALY/'
    var_file = '*'

    ds_RAW = xr.open_mfdataset(indir_data_RAW+'*'+var_file+'*'+str(year)+'.nc', chunks = {'time':1}, parallel=True)
    ds_RAW = ds_RAW.rename({'latitude':'lat', 'longitude':'lon'})

    ds_FILTER = xr.open_mfdataset(indir_data_FILTERED + '*' + str(year) + '.nc', chunks = {'time':1}, parallel=True)

    ds_VAR_TCWV = xr.open_mfdataset(indir_data_VAR + 'TCWV_'+ season + '_box_A.nc', chunks = {'lat' : 1}, parallel=True)
    ds_VAR_TOT_TCWV = xr.open_mfdataset(indir_data_VAR_TOT + 'TCWV_'+ season + '_box_A.nc', chunks = {'lat' : 1}, parallel=True)

    ds_RAW_TCWV = ds_RAW.reindex(lat=list(reversed(ds_RAW.lat)))
    ds_FILTER_TCWV = ds_FILTER.reindex(lat=list(reversed(ds_FILTER.lat)))
    """
    ############
    ### u ###
    ############
    indir_data_RAW = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/RAW_ANOMALY/u/'
    indir_data_FILTERED = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/FILTERED_ANOMALY/u/'
    var_file = 'anom_u_brut_ERA5_3H_'

    ds_RAW = xr.open_mfdataset(indir_data_RAW+'*'+var_file+'*'+str(year)+'.nc', chunks = {'time' : 1}, parallel=True)
    ds_RAW = ds_RAW.sel(level = 200)
        
    var_file = 'anom_u_z850_brut_ERA5_3H_'
    _ds_RAW = xr.open_mfdataset(indir_data_RAW+'*'+var_file+'*'+str(year)+'.nc', chunks = {'time' : 1}, parallel=True)
    
    ds_RAW = xr.concat([ds_RAW, _ds_RAW], dim = 'level')
    ds_RAW = ds_RAW.rename({'latitude':'lat', 'longitude':'lon'})

    ds_FILTER = xr.open_mfdataset(indir_data_FILTERED + '*' + str(year) + '.nc', chunks = {'time' : 1}, parallel=True)

    ds_RAW_u = ds_RAW.reindex(lat=list(reversed(ds_RAW.lat)))
    ds_FILTER_u = ds_FILTER.reindex(lat=list(reversed(ds_FILTER.lat)))
    
    ############
    ### v ###
    ############
    indir_data_RAW = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/RAW_ANOMALY/v/'
    indir_data_FILTERED = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/FILTERED_ANOMALY/v/'
    var_file = 'anom_v_brut_ERA5_3H_'

    ds_RAW = xr.open_mfdataset(indir_data_RAW+'*'+var_file+'*'+str(year)+'.nc', chunks = {'time' : 1}, parallel=True)
    ds_RAW = ds_RAW.sel(level = 200)
        
    var_file = 'anom_v_z850_brut_ERA5_3H_'
    _ds_RAW = xr.open_mfdataset(indir_data_RAW+'*'+var_file+'*'+str(year)+'.nc', chunks = {'time' : 1}, parallel=True)
    
    ds_RAW = xr.concat([ds_RAW, _ds_RAW], dim = 'level')
    ds_RAW = ds_RAW.rename({'latitude':'lat', 'longitude':'lon'})

    ds_FILTER = xr.open_mfdataset(indir_data_FILTERED + '*' + str(year) + '.nc', chunks = {'time' : 1}, parallel=True)

    ds_RAW_v = ds_RAW.reindex(lat=list(reversed(ds_RAW.lat)))
    ds_FILTER_v = ds_FILTER.reindex(lat=list(reversed(ds_FILTER.lat)))

    
    ############
    ### vo ###
    ############
    indir_data_RAW = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/RAW_ANOMALY/vo/'
    indir_data_FILTERED = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/FILTERED_ANOMALY/vo/'
    var_file = '*'

    ds_RAW = xr.open_mfdataset(indir_data_RAW+'*'+var_file+'*'+str(year)+'.nc', chunks = {'time' : 1}, parallel=True)
    ds_RAW = ds_RAW.rename({'latitude':'lat', 'longitude':'lon'})

    ds_FILTER = xr.open_mfdataset(indir_data_FILTERED + '*' + str(year) + '.nc', chunks = {'time' : 1}, parallel=True)

    ds_RAW_vo = ds_RAW.reindex(lat=list(reversed(ds_RAW.lat)))
    ds_FILTER_vo = ds_FILTER.reindex(lat=list(reversed(ds_FILTER.lat)))
    
    ############
    ### d ###
    ############
    indir_data_RAW = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/RAW_ANOMALY/d/'
    indir_data_FILTERED = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/FILTERED_ANOMALY/d/'
    var_file = '*'

    ds_RAW = xr.open_mfdataset(indir_data_RAW+'*'+var_file+'*'+str(year)+'.nc', chunks = {'time' : 1}, parallel=True)
    ds_RAW = ds_RAW.rename({'latitude':'lat', 'longitude':'lon'})

    ds_FILTER = xr.open_mfdataset(indir_data_FILTERED + '*' + str(year) + '.nc', chunks = {'time' : 1}, parallel=True)

    ds_RAW_d = ds_RAW.reindex(lat=list(reversed(ds_RAW.lat)))
    ds_FILTER_d = ds_FILTER.reindex(lat=list(reversed(ds_FILTER.lat)))
    
    ############
    ### t ###
    ############
    indir_data_RAW = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/RAW_ANOMALY/t/'
    indir_data_FILTERED = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/FILTERED_ANOMALY/t/'
    var_file = '*'

    ds_RAW = xr.open_mfdataset(indir_data_RAW+'*'+var_file+'*'+str(year)+'.nc', chunks = {'time' : 1}, parallel=True)
    ds_RAW = ds_RAW.rename({'latitude':'lat', 'longitude':'lon'})

    ds_FILTER = xr.open_mfdataset(indir_data_FILTERED + '*' + str(year) + '.nc', chunks = {'time' : 1}, parallel=True)

    ds_RAW_t = ds_RAW.reindex(lat=list(reversed(ds_RAW.lat)))
    ds_FILTER_t = ds_FILTER.reindex(lat=list(reversed(ds_FILTER.lat)))
    """
    ###########
    ### OLR ###
    ###########
    indir_data_RAW = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/RAW_ANOMALY/OLR/'
    indir_data_FILTERED = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/FILTERED_ANOMALY/OLR/'
    indir_data_VAR = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/ANALYSIS/VARIANCE/ANOMALY_FILTERED/'
    indir_data_VAR_TOT = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/ANALYSIS/VARIANCE/RAW_ANOMALY/'

    var_file = 'anom_OLR_brut_ERA5_3H'
    
    ### Open DATA
    ds_RAW_OLR = xr.open_mfdataset(indir_data_RAW+'*'+var_file+'*'+str(year)+'.nc', chunks = {'time' : 100}, parallel=True)
    ds_FILTER_OLR = xr.open_mfdataset(indir_data_FILTERED + '*' + str(year) + '.nc', chunks = {'time' : 100}, parallel=True)
    ds_VAR_OLR = xr.open_mfdataset(indir_data_VAR + 'OLR_' + season + '_box_A.nc', chunks = {'time' : 1}, parallel=True)
    ds_VAR_TOT = xr.open_mfdataset(indir_data_VAR_TOT + 'OLR_' + season + '_box_A.nc', chunks = {'time' : 1}, parallel=True)
    
    ### Reasign coords
#     ds_FILTER_TCWV = ds_FILTER_TCWV.assign_coords(lon = (((ds_FILTER_TCWV.lon + 180) % 360) - 180)).sortby('lon')
#     ds_RAW_TCWV = ds_RAW_TCWV.assign_coords(lon = (((ds_RAW_TCWV.lon + 180) % 360) - 180)).sortby('lon')

#     ds_FILTER_OLR = ds_FILTER_OLR.assign_coords(lon = (((ds_FILTER_OLR.lon + 180) % 360) - 180)).sortby('lon')
#     ds_RAW_OLR = ds_RAW_OLR.assign_coords(lon = (((ds_RAW_OLR.lon + 180) % 360) - 180)).sortby('lon')
     
        
    # concact
    ds_VAR = xr.merge([ds_VAR_TCWV, ds_VAR_OLR])
    ds1 = xr.merge([ds_FILTER_TCWV, ds_FILTER_OLR])#, ds_FILTER_u, ds_FILTER_v, ds_FILTER_t]) # , ds_FILTER_vo, ds_FILTER_d])
    ds2 = xr.merge([ds_RAW_TCWV, ds_RAW_OLR])#, ds_RAW_u, ds_RAW_v, ds_RAW_t]) # , ds_RAW_vo, ds_RAW_d])
    
    ds1 = ds1.assign_coords(lon = (((ds1.lon + 180) % 360) - 180)).sortby('lon')
    ds2 = ds2.assign_coords(lon = (((ds2.lon + 180) % 360) - 180)).sortby('lon')
    
    return ds_VAR, ds1, ds2