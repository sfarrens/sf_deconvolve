#  @file astro.py
#
#  ASTRONOMY FUNCTIONS
#
#  Basic astronomy functions
#  for conversion etc.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np


# CHECK RA AND DEC VALUES

##
#  Function that checks if the input right ascension is valid.
#
#  @param[in] ra: Input right ascension.
#
#  @exception ValueError for invalid RA.
#
def check_ra(ra):

    '''
    Function that checks if the input right ascension is valid.
    '''

    if np.any((np.array(ra) < 0.0) | (np.array(ra) > 360.0)):
        raise ValueError('Invalid RA value encountered!')


##
#  Function that checks if the input declination is valid.
#
#  @param[in] dec: Input declination.
#
#  @exception ValueError for invalid Dec.
#
def check_dec(dec):

    '''
    Function that checks if the input declination is valid.
    '''

    if np.any((np.array(dec) < -90.0) | (np.array(dec) > 90.0)):
        raise ValueError('Invalid Dec value encountered!')


##
#  Function that checks if the input object is valid.
#
#  @param[in] obj: RA and Dec of input object.
#
def check_obj(obj):

    '''
    Function that checks if the input object is valid.
    '''

    check_ra(obj[0])
    check_dec(obj[1])


# RADIANS/DEGREES CONVERSION

##
#  Function that converts angle from degrees to radians.
#
#  @param[in] angle: Input angle in degrees.
#
#  @return Output angle in radians.
#
def deg2rad(angle):

        return angle * np.pi / 180.0


##
#  Function that converts angle from radians to degrees.
#
#  @param[in] angle: Input angle in radians.
#
#  @return Output angle in degrees.
#
def rad2deg(angle):

    return angle * 180.0 / np.pi


# COORDINATE CONVERSIONS

##
#  Function that converts Equatorial Coordinates (RA,
#  Dec) to Cartesian coordinates (X, Y and Z) given a
#  distance r.
#
#  @param[in] ra: Right ascension in degrees.
#  @param[in] dec: Declination in degrees.
#  @param[in] r: Distance in given units.
#
#  @return X, Y and Z in units of r.
#
def radec2xyz(ra, dec, r):

    x = np.around(r * np.cos(deg2rad(ra)) * np.cos(deg2rad(dec)), 8)
    y = np.around(r * np.sin(deg2rad(ra)) * np.cos(deg2rad(dec)), 8)
    z = np.around(r * np.sin(deg2rad(dec)), 8)

    return x, y, z


##
#  Function that converts Cartesian coordinates (X, Y
#  and Z) to Equatorial Coordinates (RA, Dec) with a
#  distance r.
#
#  @param[in] x: X-coordinate in given units.
#  @param[in] y: Y-coordinate in given units.
#  @param[in] z: Z-coordinate in given units.
#
#  @return Ra and Dec in degrees and r in units of
#  X, Y and Z
#
def xyz2radec(x, y, z):

    r = np.around(np.sqrt(x ** 2 + y ** 2 + z ** 2), 8)
    ra = np.around(rad2deg(np.arctan2(y, x)), 8)
    dec = np.around(rad2deg(np.arcsin(z / r)), 8)

    if ra < 0.0:
        ra += 360.0

    return ra, dec, r


# ANGULAR SEPARATION

##
#  Function that calculates the angular separation
#  in degrees between to objects.
#
#  @param[in] obj1: RA and Dec of object 1 in degrees.
#  @param[in] obj2: RA and Dec of object 2 in degrees.
#
#  @return Angular separation in degrees.
#
def ang_sep(obj1, obj2):

    check_obj(obj1)
    check_obj(obj2)

    dist = np.around(np.sin(deg2rad(obj1[1])) * np.sin(deg2rad(obj2[1])) +
                     np.cos(deg2rad(obj1[1])) * np.cos(deg2rad(obj2[1])) *
                     np.cos(deg2rad(obj1[0]) - deg2rad(obj2[0])), 10)

    return rad2deg(np.array(np.arccos(dist)))


# MAGNITUDE/FLUX CONVERSION

##
#  Function that converts magnitude to flux given a
#  zero-point.
#
#  @param[in] mag: Magnitude.
#  @param[in] zero: Zero-point.
#
#  @return Flux.
#
def mag2flux(mag, zero):

    if mag > 0:
        return 10 ** ((zero - mag) / 2.5)

    else:
        return -10 ** ((zero + mag) / 2.5)


##
#  Function that converts flux to magnitude given a
#  zero-point.
#
#  @param[in] flux: Flux.
#  @param[in] zero: Zero-point.
#
#  @return Magnitude.
#
def flux2mag(flux, zero):

    if flux > 0:
        return -2.5 * np.log10(flux) + zero

    else:
        return 99.0
