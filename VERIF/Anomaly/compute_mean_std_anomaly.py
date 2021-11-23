#!/usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import h5netcdf

import xarray as xr
import pandas as pd

#from dask.distributed import Client, LocalCluster
cluster = LocalCluster(processes=False, dashboard_address = None, local_directory=tempDir, protocol = 'tcp://')
client = Client(cluster)
client

#################################################################################################
def isLeapYear (yearN):
    if ((yearN % 4 == 0) and (yearN % 100 != 0)) or (yearN % 400 == 0):
        reponse = True
    else:
        reponse = False
    print(reponse, '\n')
    return reponse

def hour_mean(x):
     return x.groupby('time.hour').mean('time')
    
def hour_sum(x):
     return x.groupby('time.hour').sum('time')
    
def hour_std(x):
     return x.groupby('time.hour').std('time')

##########################################################################################################
indir_smot = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/SMOTHED_CLIM/'
indir_anom = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/RAW_ANOMALY/TCWV/'
outdir = '/cnrm/tropics/commun/DATACOMMUN/WAVE/NO_SAVE/DATA/RAW_ANOMALY/TCWV/MEAN/'

var = 'tcwv'

############################################################################################################
ds_anom = xr.open_mfdataset(indir_anom+'*.nc', engine='h5netcdf', chunks = {'time' : 1,'latitude':100, 'longitude' : 570}, parallel=True)
ds_smot = xr.open_mfdataset(indir_smot+'*'+var+'*.nc', engine='h5netcdf', chunks = {'time' : 1,'latitude':100, 'longitude' : 570}, parallel=True)
ds_anom = ds_anom.sel(latitude = slice(40.1,-40.1))
ds_smot = ds_smot.sel(latitude = slice(40.1,-40.1))
ds_anom
ds_anom_p = ds_anom.sel(time = slice("2012","2016"))

ds_MMS = ds_anom.groupby('time.month').mean('time')
ds_MMS = ds_MMS.load()
ds_MMS.to_netcdf(outdir+'ds_anom_MM.nc', mode = 'w')
ds_MSS = ds_anom.groupby('time.season').mean('time')
ds_MSS = ds_MSS.load()
ds_MSS.to_netcdf(outdir+'ds_anom_MS.nc', mode = 'w')

ds_std_anom = ds_anom.std('time')
ds_std_anom = ds_std_anom.load()
ds_std_anom.to_netcdf(outdir+'ds_anom_SDY.nc')
ds_std_anom_M = ds_anom.groupby('time.month').std('time')
ds_std_anom_M = ds_std_anom_M.load()
ds_std_anom_M.to_netcdf(outdir+'ds_anom_STM.nc')
ds_std_anom_S = ds_anom.groupby('time.season').std('time')
ds_std_anom_S = ds_std_anom_S.load()
ds_std_anom_S.to_netcdf(outdir+'ds_anom_SSS.nc')


ds_MMS = ds_anom_p.groupby('time.month').mean('time')
ds_MMS = ds_MMS.load()
ds_MMS.to_netcdf(outdir+'ds_anom_MM_TOUCAN.nc', mode = 'w')
ds_MSS = ds_anom_p.groupby('time.season').mean('time')
ds_MSS = ds_MSS.load()
ds_MSS.to_netcdf(outdir+'ds_anom_MS_TOUCAN.nc', mode = 'w')

ds_std_anom  = ds_anom_p.std('time')
ds_std_anom = ds_std_anom.load()
ds_std_anom.to_netcdf(outdir+'ds_anom_SDY_TOUCAN.nc')
ds_std_anom_M = ds_anom_p.groupby('time.month').std('time')
ds_std_anom_M = ds_std_anom_M.load()
ds_std_anom_M.to_netcdf(outdir+'ds_anom_STM_TOUCAN.nc')
ds_std_anom_S = ds_anom_p.groupby('time.season').std('time')
ds_std_anom_S = ds_std_anom_S.load()
ds_std_anom_S.to_netcdf(outdir+'ds_anom_SSS_TOUCAN.nc')


print('end script')
