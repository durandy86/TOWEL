#!/usr/bin/env python
# coding: utf-8

import numpy as np

import xarray as xr
import xarray.ufuncs as xu
import xrft
import pandas as pd

def addNegative(ds_FILTER, ds_std_mean, ds_std, wave, da_std_TOT):
    # input : 
    # ds_FILTER = dataset of the anomaly filtered
    # ds_std* = dataset of the standard deviation for the anomaly filtered
    # wave = name
    # output : 
    # dataset with new variables 'STD_' + wave corresponding to the standard deviation
    ### Add negatif standard deviation to the DATAs
    # Mean
    ds_FILTER['STD_' + wave + '_G'] = ds_FILTER[wave] / ds_FILTER[wave]
    ds_FILTER['STD_' + wave + '_G'] = (ds_FILTER['STD_'+ wave + '_G'] * ds_std_mean[wave].values)
    ds_FILTER['STD_' + wave + '_G_N'] = -1 * ds_FILTER['STD_'+ wave + '_G'] 

    # Local
    ds_FILTER['STD_' + wave ] = ds_FILTER[wave] / ds_FILTER[wave]
    ds_FILTER['STD_' + wave ] = (ds_FILTER['STD_'+ wave] * ds_std[wave].values)
    ds_FILTER['STD_' + wave + '_N']  = -1 * ds_FILTER['STD_'+ wave] 
    
    # Anomaly Total
    ds_FILTER['STD_ano'] = da_std_TOT / da_std_TOT
    ds_FILTER['STD_ano'] = (ds_FILTER['STD_ano'] * da_std_TOT.values)
    ds_FILTER['STD_ano_N']  = -1 * ds_FILTER['STD_ano'] 
    return ds_FILTER

##########################################################################
def computeDayBelowAbove(ds_FILTER,ds_VAR,coeff, wave = 'OLR_Kelvin'):
    _da_nb_jour = ((ds_VAR[wave] * 0.) * coeff).compute()
    _da_nb_jour_N = ((ds_VAR[wave] * 0.) * coeff).compute()

    for t in range(ds_FILTER.time.size) : 
        __ds_FILTER = ds_FILTER[wave].isel(time = t).load() 
        _da_nb_jour = xr.where((__ds_FILTER >= ds_VAR[wave]) == True , _da_nb_jour + 1, _da_nb_jour)
        _da_nb_jour_N = xr.where((__ds_FILTER <= -ds_VAR[wave]) == True , _da_nb_jour_N + 1, _da_nb_jour_N)
    
    return _da_nb_jour.compute(), _da_nb_jour_N.compute()

###########################################################################
def keepData(ds_FILTER, coeff, wave, da_RAW):
    ds_below = ds_FILTER.copy()
    ds_above = ds_FILTER.copy()

    ds_below[wave] = xr.where(ds_FILTER[wave] <= (ds_FILTER['STD_' + wave +'_N'] * coeff),
                       ds_FILTER[wave], 0)
    ds_above[wave] = xr.where(ds_FILTER[wave] >= (ds_FILTER['STD_' + wave] * coeff),
                       ds_FILTER[wave], 0)
    
    ### Create new variable
    ds_FILTER[wave + '_sum_TS'] = ds_below[wave] + ds_above[wave]
    ds_FILTER[wave + '_below_TS'] = ds_below[wave]
    ds_FILTER[wave + '_above_TS'] = ds_above[wave]

    ds_FILTER[wave + '_ano_TS'] = xr.where((ds_FILTER[wave] >= (ds_FILTER['STD_' + wave] * coeff)) | (ds_FILTER[wave] <= (ds_FILTER['STD_' + wave + '_N'] * coeff)) == True,
                            da_RAW, 0)
        
    ##############
    da_below = da_RAW.copy()
    da_above = da_RAW.copy()
    
    da_below = xr.where(da_RAW <= (ds_FILTER['STD_ano'] * coeff),
                       da_RAW, 0)
    da_above = xr.where(da_RAW >= (ds_FILTER['STD_ano_N'] * coeff),
                       da_RAW, 0)
    
    ### Create new variable
    ds_FILTER['ano_sum_TS_TOT'] = da_below + da_above
    ds_FILTER['ano_below_TS_TOT'] = da_below
    ds_FILTER['ano_above_TS_TOT'] = da_above

    ds_FILTER[wave + '_ano_TS_TOT'] = xr.where((da_RAW >= (ds_FILTER['STD_ano'] * coeff)) | (da_RAW <= (ds_FILTER['STD_ano_N'] * coeff)) == True,
                            ds_FILTER[wave], 0)
    
    return ds_FILTER

######################################################################################################################
######################################################################################################################
######################################################################################################################
def computeDayBelowAbove_netCDF(ds_FILTER,ds_VAR,coeff, wave = 'OLR_Kelvin'):
    _da_nb_jour = ((ds_VAR[wave] * 0.) * coeff).compute()
    _da_nb_jour_N = ((ds_VAR[wave] * 0.) * coeff).compute()

    for t in range(ds_FILTER.time.size) : 
        __ds_FILTER = ds_FILTER[wave].isel(time = t).persist() 
        _da_nb_jour = xr.where((__ds_FILTER >= ds_VAR[wave]) == True , _da_nb_jour + 1, _da_nb_jour)
        _da_nb_jour_N = xr.where((__ds_FILTER <= -ds_VAR[wave]) == True , _da_nb_jour_N + 1, _da_nb_jour_N)
    
    ds = _da_nb_jour.to_dataset(name = 'nb_jour_Pos')
    ds['nb_jour_Neg'] = _da_nb_jour_N
    ds.to_netcdf('/cnrm/tropics/commun/DATACOMMUN/WAVE/ComputeTempo/SEUIL/' + wave + '_' + coeff + 'sigma.nc')