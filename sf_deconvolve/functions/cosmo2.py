#  @file cosmo2.py
#
#  COSMOLOGY FUNCTIONS
#
#  Cosmology routines for essential distance measures.
#
#  REFERENCES:
#  1) D.Hogg, Distance Measures in Cosmology, 2000. (H2000)
#  2) J.Peacock, Cosmological Physics, 1999. (P1999)
#
#  @author Samuel Farrens
#  @version 2.0
#  @date 2015
#

import numpy as np
from library import const
from functions.extra_math import integ_2arg, vinteg_2arg


##
#  This function checks if the cosmological
#  parameters are valid.
#
#  @param[in] z: Redshift.
#  @param[in] Omega_M: Matter density parameter.
#  @param[in] Omega_L: Dark energy density parameter.
#
#  @exception ValueError for invalid z, Omega_M or
#  Omega_L.
#
def check_cosmo(z, Omega_M, Omega_L):

    if ((np.any(z < 0.0)) | (np.any(Omega_M < 0.0)) | (np.any(Omega_L < 0.0))):
        raise ValueError('Invalid cosmology!')


##
#  This function calculates the Hubble
#  time T_H in Gyr.
#
#  Equation 3 from H2000.
#
#  @param[in] H_0: Hubble constant [km/s/Mpc].
#
#  @return Hubble time in Gyr.
#
#  @exception ValueError for invalid H_0.
#
def t_H(H_0):

    if H_0 <= 0.0:
        raise ValueError('Invalid Hubble constant value!')

    return const.MPC / (H_0 * const.YEAR * 1e9)


##
#  This function calculates the age of the
#  Universe in Gyr down to a given redshift.
#
#  Equation 5.2 from P1999.
#
#  @param[in] H_0: Hubble constant [km/s/Mpc].
#  @param[in] z: Redshift.
#  @param[in] Omega_M: Matter density parameter.
#  @param[in] Omega_L: Dark energy density parameter.
#
#  @return The age of the Universe in Gyr.
#
def age(H_0, z, Omega_M, Omega_L):

    check_cosmo(z, Omega_M, Omega_L)

    def func(z, Omega_M, Omega_L):
        A = 1.0 + z
        B = (1.0 + z) ** 2 * (1.0 + Omega_M * z)
        C = z * (2.0 + z) * Omega_L
        return 1.0 / (A * np.sqrt(B - C))

    return integ_2arg(func, z, np.inf, Omega_M, Omega_L) * t_H(H_0)


##
#  This function calculates the Hubble distance D_H
#  in Mpc.
#
#  Equation 4 from H2000.
#
#  @param[in] H_0: Hubble constant [km/s/Mpc].
#
#  @return The Hubble distance in Mpc.
#
def d_H(H_0):

    return const.C / H_0


##
#  This function calculates E(z), the Hubble
#  parameter.
#
#  Equation 14 from H2000.
#
#  @param[in] z: Redshift.
#  @param[in] Omega_M: Matter density parameter.
#  @param[in] Omega_L: Dark energy density parameter.
#
#  @return E(z) [H0 = c = 1.0].
#
def E(z, Omega_M, Omega_L):

    check_cosmo(z, Omega_M, Omega_L)

    Omega_K = 1.0 - Omega_M - Omega_L

    return np.sqrt(Omega_M * (1.0 + z) ** 3 + Omega_K *
                   (1.0 + z) ** 2 + Omega_L)


##
#  This function calculates 1.0/E(z).
#
#  @param[in] z: Redshift.
#  @param[in] Omega_M: Matter density parameter.
#  @param[in] Omega_L: Dark energy density parameter.
#
#  @return 1.0 / E(z) [H0 = c = 1.0].
#
def E_inv(z, Omega_M, Omega_L):

    return 1.0 / E(z, Omega_M, Omega_L)


##
#  This function calculates the line-of-sight comoving
#  distance D_C.
#
#  Equation 15 from H2000.
#
#  @param[in] z: Redshift.
#  @param[in] Omega_M: Matter density parameter.
#  @param[in] Omega_L: Dark energy density parameter.
#
#  @return D_C [H0 = c = 1.0].
#
def d_comov(z, Omega_M, Omega_L):

    check_cosmo(z, Omega_M, Omega_L)

    if isinstance(z, (list, tuple, np.ndarray)):
        return vinteg_2arg(E_inv, 0, z, Omega_M, Omega_L)

    else:
        return integ_2arg(E_inv, 0, z, Omega_M, Omega_L)


##
#  This function calculates the proper motion distance
#  D_M (transverse comoving distance).
#
#  Equation 16 from H2000.
#
#  @param[in] z: Redshift.
#  @param[in] Omega_M: Matter density parameter.
#  @param[in] Omega_L: Dark energy density parameter.
#
#  @return D_M [H0 = c = 1.0].
#
def d_prop(z, Omega_M, Omega_L):

    D_M = d_comov(z, Omega_M, Omega_L)

    Omega_K = 1.0 - Omega_M - Omega_L

    if Omega_K > 0:
        D_M = np.sinh(np.sqrt(np.abs(Omega_K)) * D_M) / \
          np.sqrt(np.abs(Omega_K))

    elif Omega_K < 0:
        D_M = np.sin(np.sqrt(np.abs(Omega_K)) * D_M) / \
          np.sqrt(np.abs(Omega_K))

    return D_M


##
#  This function calculates the angular diameter
#  distance D_A.
#
#  Equation 18 from H2000.
#
#  @param[in] z: Redshift.
#  @param[in] Omega_M: Matter density parameter.
#  @param[in] Omega_L: Dark energy density parameter.
#
#  @return D_A [H0 = c = 1.0].
#
def d_angdi(z, Omega_M, Omega_L):

    return d_prop(z, Omega_M, Omega_L) / (1.0 + z)


##
#  This function calculates the luminosity distance
#  D_L.
#
#  Equation 21 from H2000.
#
#  @param[in] z: Redshift.
#  @param[in] Omega_M: Matter density parameter.
#  @param[in] Omega_L: Dark energy density parameter.
#
#  @return D_L [H0 = c = 1.0].
#
def d_lum(z, Omega_M, Omega_L):

    return d_prop(z, Omega_M, Omega_L) * (1.0 + z)


##
#  This function calculates the derivative of the proper
#  motion distance with respect to redshift dD_M/dz.
#
#  @param[in] z: Redshift.
#  @param[in] Omega_M: Matter density parameter.
#  @param[in] Omega_L: Dark energy density parameter.
#
#  @return dD_M/dz [H0 = c = 1.0].
#
def dD_prop_dz(z, Omega_M, Omega_L):

    Omega_K = 1.0 - Omega_M - Omega_L

    dD_Mdz = E_inv(z, Omega_M, Omega_L)

    if Omega_K < 0:
        D_M = d_prop(z, Omega_M, Omega_L)
        ddMdz = np.sqrt(1.0 - Omega_K * D_M ** 2) * dD_Mdz

    elif Omega_K > 0:
        D_M = d_prop(z, Omega_M, Omega_L)
        ddMdz = np.sqrt(1.0 + Omega_K * D_M ** 2) * dD_Mdz

    return dD_Mdz


##
#  This function calculates the one-steradian
#  differential comoving volume dV_C/dz.
#
#  Equation 28 from H2000.
#
#  @param[in] z: Redshift.
#  @param[in] Omega_M: Matter density parameter.
#  @param[in] Omega_L: Dark energy density parameter.
#
#  @return dV_C/dz [H0 = c = 1.0].
#
def dV_comov_dz(z, Omega_M, Omega_L):

    Omega_K = 1.0 - Omega_M - Omega_L

    D_M = d_prop(z, Omega_M, Omega_L)

    dD_Mdz = dd_prop_dz(z, Omega_M, Omega_L)

    return (D_M ** 2 * dD_Mdz) / np.sqrt(1.0 + Omega_K * D_M ** 2)


##
#  This function calculates the full comoving volume
#  V_C.
#
#  Equation 29 from H2000.
#
#  @param[in] z: Redshift.
#  @param[in] Omega_M: Matter density parameter.
#  @param[in] Omega_L: Dark energy density parameter.
#
#  @return V_C [H0 = c = 1.0].
#
def v_comov(z, Omega_M, Omega_L):

    Omega_K = 1.0 - Omega_M - Omega_L

    D_M = d_prop(z, Omega_M, Omega_L)

    V_C = (4.0 * np.pi * D_M ** 3) / 3.0

    if Omega_K < 0:
        A = (2.0 * np.pi / Omega_K)
        B = D_M * np.sqrt(1.0 + Omega_K * D_M ** 2)
        C = (np.arcsin(np.sqrt(np.abs(Omega_K)) * D_M) /
             np.sqrt(np.abs(Omega_K)))
        V_C = A * (B - C)

    elif Omega_K > 0:
        A = (2.0 * np.pi / Omega_K)
        B = D_M * np.sqrt(1.0 + Omega_K * D_M ** 2)
        C = (np.arcsinh(np.sqrt(np.abs(Omega_K)) * D_M) /
             np.sqrt(np.abs(Omega_K)))
        V_C = A * (B - C)

    return V_C


##
#  This function calculates the critical
#  density rho_c.
#
#  Equation 3.25 from P1999.
#
#  @param[in] H_0: Hubble constant [km/s/Mpc].
#  @param[in] z: Redshift.
#  @param[in] Omega_M: Matter density parameter.
#  @param[in] Omega_L: Dark energy density parameter.
#
#  @return The critical density in kg/m^3.
#
def rho_crit(H_0, z, Omega_M, Omega_L):

    H2 = ((H_0 / const.MPC) * E(z, Omega_M, Omega_L)) ** 2

    return (3.0 * H2) / (8.0 * np.pi * const.G)
