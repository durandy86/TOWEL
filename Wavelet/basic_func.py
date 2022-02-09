# Audrey D.
# June 2019


import numpy as np

import xarray as xr

# Definition of some constant

deg2m = 111000
day2s = 86400


def load_simu(exp,ti,**kwargs):

    """
    Load a CROCO simulation into an xarray dataset 
    """

    options={
        'chunk_xy': False,
        'lat_min' : -25.0,
        'lat_max' :  25.0,
        'lon_min' :   0.0, 
        'lon_max' : 140.0,
        }

    options.update(kwargs)


    indir='/data/olvac/travail_en_cours/delpech/runs_datarmor/'

    data = xr.open_dataset(indir+exp+'/Eq_his_'+ti+'.nc')

    zr = get_z(data,'r')
    zr_u = 0.5*(zr[:,:,1:]+zr[:,:,:-1])
    zr_v = 0.5*(zr[:,1:,:]+zr[:,:-1,:])

    data=data.assign_coords(zr_u=(['s_rho','y_u','x_u'],zr_u))
    data=data.assign_coords(zr_v=(['s_rho','y_v','x_v'],zr_v))

    data=data.assign_coords(zr=(['s_rho','y_rho','x_rho'],zr))

    zw = get_z(data,'w')
    zw_u = 0.5*(zw[:,:,1:]+zw[:,:,:-1])
    zw_v = 0.5*(zw[:,1:,:]+zw[:,:-1,:])

    data=data.assign_coords(zw_u=(['s_w','y_u','x_u'],zw_u))
    data=data.assign_coords(zw_v=(['s_w','y_v','x_v'],zw_v))

    data=data.assign_coords(zw=(['s_w','y_rho','x_rho'],zw))

    if options['chunk_xy'] :
        print('chunk xy activated')
        data = chunk_data_xy(data,options['lat_min'],options['lat_max'],options['lon_min'],options['lon_max'])
 
    return data



def load_modes(exp,ti,mode,**kwargs):

    """
    Load a CROCO barotropic field into an xarray dataset 
    """

    options={
        'chunk_xy': False,
        'lat_min' : -25.0,
        'lat_max' :  25.0,
        'lon_min' :   0.0, 
        'lon_max' : 140.0,
        }

    options.update(kwargs)

    indir='/data/olvac/travail_en_cours/delpech/runs_datarmor/'

    data = xr.open_dataset(indir+exp+'/Eq_'+mode+'_'+ti+'.nc')

    if options['chunk_xy'] :
        data = chunk_data_xy(data,options['lat_min'],options['lat_max'],options['lon_min'],options['lon_max'])
    
    return data


def chunk_data_xy(data,lat_min,lat_max,lon_min,lon_max):

    """
    Select only a chunk of the data delimited by lat_min, lat_max ...etc.
    """
    if 'nav_lon_u' in list(data.coords):
        x_u_min = np.where((data.nav_lon_u[0,:]>lon_min) & (data.nav_lon_u[0,:]<lon_max))[0].min() 
        x_u_max = np.where((data.nav_lon_u[0,:]>lon_min) & (data.nav_lon_u[0,:]<lon_max))[0].max() 
        y_u_min = np.where((data.nav_lat_u[:,0]>lat_min) & (data.nav_lat_u[:,0]<lat_max))[0].min()
        y_u_max = np.where((data.nav_lat_u[:,0]>lat_min) & (data.nav_lat_u[:,0]<lat_max))[0].max() 
   
    if 'nav_lon_v' in list(data.coords):
        y_v_min = np.where((data.nav_lat_v[:,0]>lat_min) & (data.nav_lat_v[:,0]<lat_max))[0].min()
        y_v_max = np.where((data.nav_lat_v[:,0]>lat_min) & (data.nav_lat_v[:,0]<lat_max))[0].max()
        x_v_min = np.where((data.nav_lon_v[0,:]>lon_min) & (data.nav_lon_v[0,:]<lon_max))[0].min()
        x_v_max = np.where((data.nav_lon_v[0,:]>lon_min) & (data.nav_lon_v[0,:]<lon_max))[0].max()
    
    if 'nav_lon_rho' in list(data.coords):
        x_r_min = np.where((data.nav_lon_rho[0,:]>lon_min) & (data.nav_lon_rho[0,:]<lon_max))[0].min()
        x_r_max = np.where((data.nav_lon_rho[0,:]>lon_min) & (data.nav_lon_rho[0,:]<lon_max))[0].max() 
        y_r_min = np.where((data.nav_lat_rho[:,0]>lat_min) & (data.nav_lat_rho[:,0]<lat_max))[0].min()
        y_r_max = np.where((data.nav_lat_rho[:,0]>lat_min) & (data.nav_lat_rho[:,0]<lat_max))[0].max()


    data_var=[]
    for var in [x for x in list(data.variables) if x not in list(data.coords)]:
        
        if 'x_v' in list(data[var].dims): 
            data_tmp = data[var].where((data[var].x_v>x_v_min) & (data[var].x_v<x_v_max),drop=True).where((data[var].y_v>y_v_min) & (data[var].y_v<y_v_max),drop=True)
            data_tmp.name = var
            data_var.append(data_tmp)
        
        if 'x_u' in list(data[var].dims):
            data_tmp = data[var].where((data[var].x_u>x_u_min) & (data[var].x_u<x_u_max),drop=True).where((data[var].y_u>y_u_min) & (data[var].y_u<y_u_max),drop=True)
            data_tmp.name = var
            data_var.append(data_tmp)

        if 'x_rho' in list(data[var].dims):
            data_tmp = data[var].where((data[var].x_rho>x_r_min) & (data[var].x_rho<x_r_max),drop=True).where((data[var].y_rho>y_r_min) & (data[var].y_rho<y_r_max),drop=True)
            data_tmp.name = var
            data_var.append(data_tmp)


    data = xr.merge(data_var)

    return data




def load_diagUV(exp,ti):

    """
    Load a CROCO momentum diag into an xarray dataset 
    """

    indir='/data/olvac/travail_en_cours/delpech/runs_datarmor/'

    data = xr.open_dataset(indir+exp+'/Eq_diagUV_'+ti+'.nc')

    #zr = get_z(data)
    #zr_u = 0.5*(zr[:,:,1:]+zr[:,:,:-1])
    #zr_v = 0.5*(zr[:,1:,:]+zr[:,:-1,:])

    #data=data.assign_coords(zr_u=(['s_rho','y_u','x_u'],zr_u))
    #data=data.assign_coords(zr_v=(['s_rho','y_v','x_v'],zr_v))

    #data=data.assign_coords(zr=(['s_rho','y_rho','x_rho'],zr))

    return data



def get_depth(h,zeta,Cs_w,sc_w,Cs_r,sc_r,hc,N,option):
    
    """
    Arguments : h (bathymetry), sc (stretching coordinate coefficient),
    N (number of vertical levels), Cs (sigma-coordinate), hc (averaged depth croco.in)

    """
    #zeta = np.nanmean(zeta,axis=0)
    if option=='w':
        cff1 = Cs_w
        sc = sc_w
    else:
        cff1 = Cs_r
        sc = sc_r
    h2 = h + hc
    cff = hc*sc
    h2inv = 1/h2

    if option=='w':

        z = np.zeros((N+1,np.shape(h)[0],np.shape(h)[1]))

        for k in range(N+1):
            z0 = cff[k] + cff1[k]*h
            z[k,:,:] = z0*(h/h2) + zeta*(1+(z0*h2inv))

    else:
        z = np.zeros((N,np.shape(h)[0],np.shape(h)[1]))
        for k in range(N):
            z0 = cff[k] + cff1[k]*h
            z[k,:,:] = z0*(h/h2) + zeta*(1+(z0*h2inv))
        #z = np.transpose(z,axes=(1,2,0))


    return z





def scoord2z(point_type, ssh, h, theta_s, theta_b, hc, scoord, N):
    """
    scoord2z finds z at either rho or w points (positive up, zero at rest surface)
    h          = array of depths (e.g., from grd file)
    theta_s    = surface focusing parameter
    theta_b    = bottom focusing parameter
    hc         = critical depth
    N          = number of vertical rho-points
    point_type = 'r' or 'w'
    scoord     = 'new2008' :new scoord 2008, 'new2006' : new scoord 2006,
                  or 'old1994' for Song scoord
    ssh       = sea surface height
    message    = set to False if don't want message

    note : routine from S. Le Gentil. 
    """
    def CSF(sc, theta_s, theta_b):
        '''
        Allows use of theta_b > 0 (July 2009)
        '''
        one64 = np.float64(1)

        if theta_s > 0.:
            csrf = ((one64 - np.cosh(theta_s * sc))
                       / (np.cosh(theta_s) - one64))
        else:
            csrf = -sc ** 2
        sc1 = csrf + one64
        if theta_b > 0.:
            Cs = ((np.exp(theta_b * sc1) - one64)
                / (np.exp(theta_b) - one64) - one64)
        else:
            Cs = csrf
        return Cs


    sc_w = (np.arange(N + 1, dtype=np.float64) - N) / N
    sc_r = ((np.arange(1, N + 1, dtype=np.float64)) - N - 0.5) / N

    if 'w' in point_type:
        sc = sc_w
        N += 1. # add a level
    else:
        sc = sc_r

    z  = np.empty((int(N),) + h.shape, dtype=np.float64)
    if scoord == 2:
        Cs = CSF(sc, theta_s, theta_b)
    else:
        try:
            cff1=1./sinh(theta_s)
            cff2=0.5/tanh(0.5*theta_s)
        except:
            cff1=0.
            cff2=0.
        Cs=(1.-theta_b)*cff1*sinh(theta_s*sc) \
            +theta_b*(cff2*tanh(theta_s*(sc+0.5))-0.5)

    if scoord == 2:
        hinv = 1. / (h + hc)
        cff = (hc * sc).squeeze()
        cff1 = (Cs).squeeze()
        for k in np.arange(N, dtype=int):
            z[k] = ssh + (ssh + h) * (cff[k] + cff1[k] * h) * hinv
    elif scoord == 1:
        hinv = 1. / h
        cff  = (hc * (sc[:] - Cs[:])).squeeze()
        cff1 = Cs.squeeze()
        cff2 = (sc + 1).squeeze()
        for k in np.arange(N, dtype=int) + 1:
            z0      = cff[k-1] + cff1[k-1] * h
            z[k-1, :] = z0 + ssh * (1. + z0 * hinv)
    else:
        raise Exception("Unknown scoord, should be 1 or 2")
    return z.squeeze()



def get_z(data,option):

    """
    Get z-levels from CROCO data as a function 
    of x, y and s.
    """
    
    # load parameters
    theta_s = data.theta_s.values.squeeze()
    theta_b = data.theta_b.values.squeeze()
    hc = data.hc.values.squeeze()
    N = data.s_rho.shape[0]
    h = data.h.values

    # time-average of ssh
    ssh = np.nanmean(data.ssh.values,0)
    
    # type of sigma-coordinate
    scoord  = data.Vtransform.values.squeeze()
    
    z_r = scoord2z('r', ssh, h, theta_s, theta_b, hc, scoord, N)
    z_w = scoord2z('w', ssh, h, theta_s, theta_b, hc, scoord, N)
        
    if option=='r':
        return z_r
    if option=='w':
        return z_w


def get_z_time(data):

    """
    Get z-levels from CROCO data as a function of
    t, x, y and s.
    """

    theta_s = data.theta_s.values.squeeze()
    theta_b = data.theta_b.values.squeeze()
    hc = data.hc.values.squeeze()
    scoord  = data.Vtransform.values.squeeze()
    N = data.s_rho.shape[0]
    h = data.h.values
    
    z_r,z_w=[],[]
    for t in range(data.time_counter.size):

        ssh = data.ssh.values[t]
        z_r.append(scoord2z('r', ssh, h, theta_s, theta_b, hc, scoord, N))
        z_w.append(scoord2z('w', ssh, h, theta_s, theta_b, hc, scoord, N))

    z_r=np.array(z_r)
        
    return z_r



