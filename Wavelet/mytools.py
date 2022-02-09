# Audrey D.
# September 2018


import sys
sys.path.append('/home/olvac/delpech/Documents/')

import numpy as np



def round_n(a,n):

    return np.round(a/n)*n



def find_nearest(array, value):

    """
    Find item in array which is the closest to 
    a given values.
    """

    array = np.asarray(array)

    idx = (np.abs(array - value)).argmin()

    return array[idx]



def find_nearest_above(my_array, target):

    """
    """

    diff = my_array - target
    mask = np.ma.less_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        return None # returns None if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return my_array[masked_diff.argmin()]



def find_nearest_below(my_array, target):

    """
    """

    diff = my_array - target
    mask = np.ma.greater_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        return None # returns None if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return my_array[masked_diff.argmax()]




def correct_longitude(longitude):

    """
    Function that correct the lon from negative values
    to > 180. Apply to a vector
    """

    longitude_corrected = np.where(longitude<0,180+(180-abs(longitude)),longitude)

    return longitude_corrected


def decorr_longitude(longitude):

    longitude_decorrect = np.where(longitude>180,longitude-2*180,longitude)

    return longitude_decorrect





def my_argmax(vect,ax):

    if np.all(np.isnan(vect)):
        return np.argmax(vect,ax)
    else : 
        return np.nanargmax(vect,ax)

def my_nanmax(vect,ax):

    if np.all(np.isnan(vect)):
        return np.nan
    else : 
        return np.nanmax(vect,ax)













