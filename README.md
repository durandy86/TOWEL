# TOWEL

## Environnement conda :
conda create -n towel -c conda-forge python dask dask-jobqueue \
            xarray zarr netcdf4 python-graphviz h5netcdf\
            jupyterlab ipywidgets \
            cartopy scikit-learn seaborn \
            hvplot \
            intake-xarray \
            xrft wrf-python \
            pyinterp

## Script :

### TROPICS_CLIM_BRUTE.ipynb 
Correspond à la climatologie brute de ERA5

