import numpy as np
from numba import njit, prange
import os
import time
import bias_parameters_lya as biaspars
import astropy.io.fits as fits
import healpy
import astropy.constants as const
from astropy.table import Table
from multiprocessing import Pool, cpu_count
import input_params as inpars
import math
import h5py
import argparse
import configparser

# **********************************************
# **********************************************
# **********************************************
# INPUT PARAMETERS
argslist=None

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='config filename')
args = parser.parse_args(argslist)

config = configparser.ConfigParser()
config.read(args.config)

nreal = int(config['SETUP']['seed'])
version = config['SETUP']['version']

# SETUP                                                                                                                                                                                                    
ngrid = int(config['SETUP']['ngrid'])
lbox = float(config['SETUP']['lbox'])
zmin = float(config['SETUP']['zmin'])
zmax = float(config['SETUP']['zmax'])

# I/O                                                                                                                                                                                                      
input_dir = config['IO']['input_dir'] + 'mock_%d/' %nreal
output_dir = config['IO']['output_dir'] + version + '/skewers-%d/' %nreal
output_aux_dir = output_dir + 'aux/'

# General parameters
posx_qso_filename = output_aux_dir + 'posxtr_zspace.dat'
posy_qso_filename = output_aux_dir + 'posytr_zspace.dat'
posz_qso_filename = output_aux_dir + 'posztr_zspace.dat'

fits_filename = output_dir + 'master.fits'

delta_filename =  input_dir + config['IO']['dm_filename']
velx_filename =  input_dir + config['IO']['vx_filename']
vely_filename =  input_dir + config['IO']['vx_filename']
velz_filename =  input_dir + config['IO']['vy_filename']

zarr_filename = config['IO']['zarr_filename']
darr_filename = config['IO']['darr_filename']

nside = 16
pixmax = 3072

num_processes = 64

# General parameters

zmin_extr = zmin

hrbinw = 0.1

lammin = 3469.9
lammax = 6500.1
dlam = 0.2

# Subgrid parameters
alpha_fgpa = 1.65

z0 = 3.0
k1 = 0.1#0.0341
nn = 0.732

normrand = 1.0
A0_1 = 1.9
A1_1 = 0.276
A0_2 = 1.25
A1_2 = 4.5 # Fix as from LyaColore

bb = 1.0
beta = 1.
bv = 1.0

# Rest-frame frequencies
lam_lya = 1215.67
lam_lyb = 1025.72
lam_SiII_1260 = 1260.42
lam_SiIII_1207 = 1206.50
lam_SiII_1193 = 1193.29
lam_SiII_1190 = 1190.42
lam_CIV =  1549.06 

# Absorption strenghts - Lya by definition is 1 (Numbers from Tab. 2 of Farr et al. 2020, https://arxiv.org/abs/1912.02763)
Ax_lya = 1.
Ax_lyb = 0.1901
Ax_SiII_1260 = 3.542e-4
Ax_SiIII_1207 = 1.8919e-3
Ax_SiII_1193 = 9.0776e-4
Ax_SiII_1190 = 1.28478e-4
Ax_CIV = 1e-4

# Observer positions
obspos = [5000.,5000.,5000.]

# Cosmological parameters (Abacus)
h = float(config['COSMOLOGY']['h'])
Om = float(config['COSMOLOGY']['Om'])
Orad = float(config['COSMOLOGY']['Orad'])
Ok = float(config['COSMOLOGY']['Ok'])
N_eff = float(config['COSMOLOGY']['N_eff'])
w_eos = float(config['COSMOLOGY']['w_eos'])
Ol = 1-Om-Ok-Orad

# Random seed for stochasticity reproducibility
np.random.seed(123456)

# Numerics
smallnum = 1e-6

# Convergence
saturate_flux = True

# Fundamental constants (not used unless one explicitly wants a Doppler broadening)
K_BOLTZMANN = 1.380649e-23        # J / K
M_PROTON = 1.67262192369e-27      # kg
C_LIGHT_KMS = 299792.458          # km / s
INV_SQRT_PI = 0.5641895835477563  # 1/sqrt(pi)

# **********************************************
# **********************************************
# **********************************************
def fftr2c(arr):
    arr = np.fft.rfftn(arr, norm='ortho')

    return arr

def fftc2r(arr):
    arr = np.fft.irfftn(arr, norm='ortho')

    return arr

# **********************************************
@njit(parallel=False, fastmath=True, cache=True)
def extract_skewers(posx, posy, posz, zmin, zmax, zarr, darr, hrbinw, delta, vx, vy, vz, ngrid, lbox, xobs, yobs, zobs):

    lcell = lbox / ngrid

    # Initialize a matrix of skewers                                                                                                                            
    # (NxM), where N=number of QSO, M=number of bins per spectrum                                                                                               

    # Let's first determine maximum and minimum distance, given the redshift range                                                                              
    dmax = np.interp(zmax, zarr, darr)
    dmin = np.interp(zmin, zarr, darr)

    # Determine the number of bins in the spectra                                                                                                               
    nbins = int((dmax-dmin) / hrbinw)
    if nbins%2>0:
        nbins-=1

    # Allocate the matrix                                                                                                                                       
    skmat = np.zeros((len(posx), nbins))
    skmat_vel = np.zeros((len(posx), nbins))
    indmat = np.zeros((len(posx), nbins))

    # Set up a template of distances                                                                                                                            
    dtemplate = np.linspace(dmin, dmax, nbins+1)
    dtemplate = 0.5*(dtemplate[1:] + dtemplate[:-1])

    ztemplate = np.interp(dtemplate, darr, zarr)

    for ii in range(len(posx)):

        ra, dec, zz = cartesian_to_sky(posx[ii],posy[ii],posz[ii], zarr, darr, xobs, yobs, zobs)

        for jj in range(len(dtemplate)):

            ztemp = ztemplate[jj]

            if ztemp<=zz and ztemp>zmin_extr and ztemp<zmax:
            
                posxlya, posylya, poszlya = sky_to_cartesian(ra,dec,ztemp, zarr, darr, xobs, yobs, zobs)
                indx = int(posxlya/lcell)
                indy = int(posylya/lcell)
                indz = int(poszlya/lcell)

                ind3d = int(indx + ngrid*(indy + ngrid*indz))

                skmat[ii,jj] = delta[indx,indy,indz]
                indmat[ii,jj] = ind3d

                vxtmp = vx[indx,indy,indz] #trilininterp(xtmp, ytmp, ztmp, vx, lbox, ngrid)
                vytmp = vy[indx,indy,indz] #trilininterp(xtmp, ytmp, ztmp, vy, lbox, ngrid)
                vztmp = vz[indx,indy,indz] #trilininterp(xtmp, ytmp, ztmp, vz, lbox, ngrid)

                sigma_rsd = bb*(1. + delta[indx,indy,indz])**beta
                
                vxrand = np.random.normal(0,sigma_rsd)
                vyrand = np.random.normal(0,sigma_rsd)
                vzrand = np.random.normal(0,sigma_rsd)

                vxtmp += vxrand
                vytmp += vyrand
                vztmp += vzrand

                # Go from cartesian to sky coordinates
                vxtmp, vytmp, vztmp, vsign = project_vector_los(posxlya, posylya, poszlya, vxtmp, vytmp, vztmp, zarr, darr, xobs, yobs, zobs)

                skmat_vel[ii,jj] = vsign#*np.sqrt(vxtmp**2 + vytmp**2 + vztmp**2)

            else:
                skmat[ii,jj] = 0.
                skmat_vel[ii,jj] = 0.
                indmat[ii,jj] = -99

    return ztemplate, dtemplate, skmat, skmat_vel, indmat

# **********************************************
@njit(parallel=False, fastmath=True, cache=True)
def sky_to_cartesian(ra,dec,zz, zarr, darr, xobs, yobs, zobs):

    dd = np.interp(zz, zarr, darr)

    ra = ra / 180. * np.pi
    dec = dec / 180. * np.pi

    posx = dd * np.cos(dec) * np.cos(ra)
    posy = dd * np.cos(dec) * np.sin(ra)
    posz = dd * np.sin(dec)

    posx += xobs
    posy += yobs
    posz += zobs

    return posx, posy, posz

# **********************************************
@njit(parallel=True, fastmath=True, cache=True)
def cartesian_to_sky_loop(posx, posy, posz, zarr, darr, xobs, yobs, zobs):

    ra = np.zeros(len(posx))
    dec = np.zeros(len(posx))
    zz = np.zeros(len(posx))

    for ii in range(len(posx)):

        possx = posx[ii] - xobs
        possy = posy[ii] - yobs
        possz = posz[ii] - zobs

        dd = np.sqrt(possx**2 + possy**2 + possz**2)
        zz[ii] = np.interp(dd, darr, zarr)
        #dec = np.arccos(posz / dd) / np.pi * 180.

        #if np.sin(dec)!=0:
        #    ra = np.arccos( posx / (dd * np.sin(dec))) / np.pi * 180.
        #else:
        #    ra = 0.
        ss = np.hypot(possx, possy)
        ra[ii] = np.arctan2(possy, possx) / np.pi * 180.
        #dec = np.arcsin(posz/dd) / np.pi * 180.
        dec[ii] = np.arctan2(possz, ss) / np.pi * 180.

        # convert to degrees
        #lon = da.rad2deg(lon)
        #lat = da.rad2deg(lat)
        #ra = np.mod(ra-360., 360.)

    return ra, dec, zz

# **********************************************                                                                                                                      
@njit(parallel=False, fastmath=True, cache=True)
def cartesian_to_sky(posx, posy, posz, zarr, darr, xobs, yobs, zobs):

    posx -= xobs
    posy -= yobs
    posz -= zobs

    dd = np.sqrt(posx**2 + posy**2 + posz**2)
    zz = np.interp(dd, darr, zarr)
    #dec = np.arccos(posz / dd) / np.pi * 180.                                                                                                                        

    #if np.sin(dec)!=0:                                                                                                                                               
    #    ra = np.arccos( posx / (dd * np.sin(dec))) / np.pi * 180.                                                                                                    
    #else:                                                                                                                                                            
    #    ra = 0.                                                                                                                                                      
    ss = np.hypot(posx, posy)
    ra = np.arctan2(posy, posx) / np.pi * 180.
    #dec = np.arcsin(posz/dd) / np.pi * 180.                                                                                                                          
    dec = np.arctan2(posz, ss) / np.pi * 180.
    # convert to degrees                                                                                                                                             
    #lon = da.rad2deg(lon)                                                                                                                                           
    #lat = da.rad2deg(lat)                                                                                                                                           
    #ra = np.mod(ra-360., 360.)                                                                                                                                      

    return ra, dec, zz

# **********************************************

@njit(parallel=False, fastmath=True, cache=True)
def project_vector_los(posx, posy, posz, vecx, vecy, vecz, zarr, darr, xobs, yobs, zobs):

    # Determine the line of sight angles ra0 and dec0
    ra0, dec0, zz0 = cartesian_to_sky(posx, posy, posz, zarr, darr, xobs, yobs, zobs)

    ra0 = ra0 / 180. * np.pi
    dec0 = dec0 / 180. * np.pi
    
    # Find the l.o.s. unit vector
    versx = np.cos(dec0) * np.cos(ra0)
    versy = np.cos(dec0) * np.sin(ra0)
    versz = np.sin(dec0)
    
    # Project the velocity vector along the l.o.s. direction
    norm = vecx*versx + vecy*versy + vecz*versz
    
    vecx = norm * versx
    vecy = norm * versy
    vecz = norm * versz

    return vecx, vecy, vecz, norm

# **********************************************
@njit(parallel=False, cache=True, fastmath=True)
def trilininterp(xx, yy, zz, arrin, lbox, ngrid):

    lcell = lbox/ngrid

    indxc = int(xx/lcell)
    indyc = int(yy/lcell)
    indzc = int(zz/lcell)

    wxc = xx/lcell - indxc
    wyc = yy/lcell - indyc
    wzc = zz/lcell - indzc

    if wxc <=0.5:
        indxl = indxc - 1
        if indxl<0:
            indxl += ngrid
        wxc += 0.5
        wxl = 1 - wxc
    elif wxc >0.5:
        indxl = indxc + 1
        if indxl>=ngrid:
            indxl -= ngrid
        wxl = 1 - wxc

    if wyc <=0.5:
        indyl = indyc - 1
        if indyl<0:
            indyl += ngrid
        wyc += 0.5
        wyl = 1 - wyc
    elif wyc >0.5:
        indyl = indyc + 1
        if indyl>=ngrid:
            indyl -= ngrid
        wyl = 1 - wyc

    if wzc <=0.5:
        indzl = indzc - 1
        if indzl<0:
            indzl += ngrid
        wzc += 0.5
        wzl = 1 - wzc
    elif wzc >0.5:
        indzl = indzc + 1
        if indzl>=0:
            indzl -= ngrid
        wzl = 1 - wzc

    wtot = wxc*wyc*wzc + wxl*wyc*wzc + wxc*wyl*wzc + wxc*wyc*wzl + wxl*wyl*wzc + wxl*wyc*wzl + wxc*wyl*wzl + wxl*wyl*wzl

    out = 0.

    out += arrin[indxc,indyc,indzc] * wxc*wyc*wzc
    out += arrin[indxl,indyc,indzc] * wxl*wyc*wzc
    out += arrin[indxc,indyl,indzc] * wxc*wyl*wzc
    out += arrin[indxc,indyc,indzl] * wxc*wyc*wzl
    out += arrin[indxl,indyl,indzc] * wxl*wyl*wzc
    out += arrin[indxc,indyl,indzl] * wxc*wyl*wzl
    out += arrin[indxl,indyc,indzl] * wxl*wyc*wzl
    out += arrin[indxl,indyl,indzl] * wxl*wyl*wzl

    return out

# **********************************************
@njit(parallel=True, cache=True, fastmath=True)
def flux_to_tau(ngrid, lbox, skmat):

    for ii in prange(skmat.shape[0]):
        for jj in range(skmat.shape[1]):

            skmat[ii,jj] = -np.log10(skmat[ii,jj])

    return skmat

# **********************************************     
@njit(parallel=True, cache=True, fastmath=True)
def tau_to_flux(ngrid, lbox, skmat):

    for	ii in prange(skmat.shape[0]):
        for jj in range(skmat.shape[1]):

            skmat[ii,jj] = np.exp(-skmat[ii,jj])

            if saturate_flux == True:
                if skmat[ii,jj]>1.:
                    skmat[ii,jj] = 1.

    return skmat

# ***************************************
@njit(cache=True)
def _gaussian_profile_scalar(du, b):
    x = du / b
    return math.exp(-x * x) * INV_SQRT_PI / b


@njit(cache=True)
def _voigt_hjerting_tepper_garcia(x, a):
    """
    Tepper-Garcia (2006, MNRAS 369, 2025) approximation to the
    Voigt-Hjerting function H(a,x), accurate to <~1% for a << 1 (always
    true for Lya: a = gamma_lorentz/b ~ 1e-3). Reduces to exp(-x^2) as
    a -> 0.
    """
    x2 = x * x
    H0 = math.exp(-x2)
    if a == 0.0:
        return H0
    if x2 < 1e-12:
        x2 = 1e-12
    Q = 1.5 / x2
    return H0 - (a * INV_SQRT_PI / x2) * (
        H0 * H0 * (4.0 * x2 * x2 + 7.0 * x2 + 4.0 + Q) - Q - 1.0
    )
 
 
@njit(cache=True)
def _voigt_profile_scalar(du, b, gamma_lorentz):
    x = du / b
    a = gamma_lorentz / b
    return _voigt_hjerting_tepper_garcia(x, a) * INV_SQRT_PI / b

njit(parallel=True, cache=True)
def _gaussian_profile_loop(du, b):
    N = du.shape[0]
    out = np.empty(N)
    for i in prange(N):
        out[i] = _gaussian_profile_scalar(du[i], b[i])
    return out
 
 
@njit(parallel=True, cache=True)
def _voigt_profile_loop(du, b, gamma_lorentz):
    N = du.shape[0]
    out = np.empty(N)
    for i in prange(N):
        out[i] = _voigt_profile_scalar(du[i], b[i], gamma_lorentz)
    return out
 
 
def gaussian_profile(du: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalized (in velocity, unit area) Gaussian line profile, area=1."""
    du = np.ascontiguousarray(du, dtype=np.float64)
    b = np.ascontiguousarray(np.broadcast_to(b, du.shape), dtype=np.float64)
    return _gaussian_profile_loop(du, b)
 
 
def voigt_profile(du: np.ndarray, b: np.ndarray, gamma_lorentz: float) -> np.ndarray:
    """
    Normalized Voigt profile (area=1 in du), via the Tepper-Garcia (2006)
    approximation, with Lorentzian HWHM `gamma_lorentz` in km/s (natural +
    collisional damping of the transition; for Lya, this is generally tiny
    compared to b except deep in damped-wing/DLA regimes).
    """
    du = np.ascontiguousarray(du, dtype=np.float64)
    b = np.ascontiguousarray(np.broadcast_to(b, du.shape), dtype=np.float64)
    return _voigt_profile_loop(du, b, float(gamma_lorentz))



@njit(parallel=True, cache=True)
def _rsd_convolution_loop(u_grid, tau_real, u_shifted, b_doppler, du_cell,
                           gamma_lorentz, periodic, L_box):
    """
    Explicit double loop over (redshift-space pixel i, real-space source
    pixel j), parallelized over i with prange:
 
        tau_s[i] = sum_j tau_real[j] * phi(u_grid[i] - u_shifted[j]; b[j]) * du_cell
    """
    N = u_grid.shape[0]
    tau_redshift = np.empty(N)
 
    for i in prange(N):
        ui = u_grid[i]
        acc = 0.0
        for j in range(N):
            du = ui - u_shifted[j]
            if periodic:
                du -= L_box * math.floor(du / L_box + 0.5)  # minimum image
            b = b_doppler[j]
            if gamma_lorentz > 0.0:
                phi = _voigt_profile_scalar(du, b, gamma_lorentz)
            else:
                phi = _gaussian_profile_scalar(du, b)
            acc += phi * tau_real[j]
        tau_redshift[i] = acc * du_cell
 
    return tau_redshift

@njit(parallel=True, cache=True)
def _rsd_convolution_windowed_loop(u_grid, tau_real, u_shifted, b_doppler, du_cell, gamma_lorentz, periodic, L_box, half_width_pixels):

    print('Full Doppler')
    
    N = u_grid.shape[0]
    tau_redshift = np.zeros(N)

    for i in prange(N):
        ui = u_grid[i]
        acc = 0.0
        for dj in range(-half_width_pixels, half_width_pixels + 1):
            j = i + dj
            if periodic:
                j = j % N
            elif j < 0 or j >= N:
                continue
            du = ui - u_shifted[j]
            if periodic:
                du -= L_box * math.floor(du / L_box + 0.5)  # minimum image
            b = b_doppler[j]
            if gamma_lorentz > 0.0:
                phi = _voigt_profile_scalar(du, b, gamma_lorentz)
            else:
                phi = _gaussian_profile_scalar(du, b)
            acc += phi * tau_real[j]
        tau_redshift[i] = acc * du_cell

    return tau_redshift

"""
def apply_rsd_to_skewer(
    u_grid: np.ndarray,
    tau_real: np.ndarray,
    v_pec: np.ndarray,
    b_doppler: np.ndarray,
    du_cell: float | None = None,
    gamma_lorentz: float = 0.0,
    periodic: bool = True,
    truncate_sigma: float | None = 8.0,
) -> np.ndarray:

    u_grid = np.ascontiguousarray(u_grid, dtype=np.float64)
    tau_real = np.ascontiguousarray(tau_real, dtype=np.float64)
    v_pec = np.ascontiguousarray(v_pec, dtype=np.float64)
    b_doppler = np.ascontiguousarray(b_doppler, dtype=np.float64)

    N = u_grid.size
    if du_cell is None:
        du_cell = float(u_grid[1] - u_grid[0])
    L_box = N * du_cell

    u_shifted = u_grid + v_pec  # shifted center of each real-space cell's contribution

    if truncate_sigma is not None:
        W = float(truncate_sigma) * float(np.max(b_doppler))
        vmax = float(np.max(np.abs(v_pec))) if N > 0 else 0.0
        half_width_pixels = int(math.ceil((W + vmax) / du_cell))
        if 2 * half_width_pixels + 1 < N:
            return _rsd_convolution_windowed_loop(
                u_grid, tau_real, u_shifted, b_doppler,
                float(du_cell), float(gamma_lorentz), bool(periodic), float(L_box),
                half_width_pixels,
            )
        # window would cover (almost) the whole skewer anyway -> exact loop is just as cheap

    return _rsd_convolution_loop(
        u_grid, tau_real, u_shifted, b_doppler,
        float(du_cell), float(gamma_lorentz), bool(periodic), float(L_box),
    )
"""

@njit(cache=True)
def _rsd_delta_kernel_deposit(u_grid, tau_real, u_shifted, du_cell, periodic, L_box):

    #print('Dirac delta')

    N = u_grid.shape[0]
    tau_redshift = np.zeros(N)
    u0 = u_grid[0]
 
    for j in range(N):
        pos = (u_shifted[j] - u0) / du_cell
        i0 = int(math.floor(pos))
        frac = pos - i0          # in [0, 1): fraction of tau going to cell i0+1
        w0 = 1.0 - frac
        w1 = frac
        i1 = i0 + 1
        tj = tau_real[j]
 
        if periodic:
            tau_redshift[i0 % N] += w0 * tj
            tau_redshift[i1 % N] += w1 * tj
        else:
            if 0 <= i0 < N:
                tau_redshift[i0] += w0 * tj
            if 0 <= i1 < N:
                tau_redshift[i1] += w1 * tj
 
    return tau_redshift

@njit(cache=True)
def _rsd_delta_kernel_deposit_comoving(
    d_grid,
    tau_real,
    d_shifted,
    dd_cell,
    periodic,
    L_box
):

    N = d_grid.shape[0]
    tau_redshift = np.zeros(N)

    d0 = d_grid[0]

    for j in range(N):

        # Position of shifted element in units of grid cells
        pos = (d_shifted[j] - d0) / dd_cell

        i0 = int(math.floor(pos))
        frac = pos - i0

        w0 = 1.0 - frac
        w1 = frac

        i1 = i0 + 1

        tj = tau_real[j]

        if periodic:

            tau_redshift[i0 % N] += w0 * tj
            tau_redshift[i1 % N] += w1 * tj

        else:

            if 0 <= i0 < N:
                tau_redshift[i0] += w0 * tj

            if 0 <= i1 < N:
                tau_redshift[i1] += w1 * tj

    return tau_redshift

def apply_rsd(
    d_grid,
    z_template,
    skmat_tau_real,
    skmat_vpec,
    b_doppler,
    periodic=True
):

    d_grid = np.asarray(d_grid, dtype=np.float64)
    z_template = np.asarray(z_template, dtype=np.float64)

    skmat_tau_real = np.asarray(
        skmat_tau_real,
        dtype=np.float64
    )

    skmat_vpec = np.asarray(
        skmat_vpec,
        dtype=np.float64
    )

    N = d_grid.size

    # d_grid is uniformly spaced in Mpc/h
    dd_cell = float(d_grid[1] - d_grid[0])

    L_box = N * dd_cell

    # Dimensionless Hubble function E(z)
    Ez = np.sqrt(
        Om * (1.0 + z_template)**3
        + Ol
    )

    # RSD displacement in Mpc/h
    #
    # dr = v / (a H) in Mpc
    # multiply by h -> Mpc/h
    #
    # H(z) = 100 h E(z)
    #
    # therefore:
    #
    # dr[Mpc/h] = v[km/s] * (1+z) / (100 E(z))
    #
    displacement = (
        skmat_vpec
        * (1.0 + z_template[None, :])
        / (100.0 * Ez[None, :])
    )

    skmat_tau_redshift = np.empty_like(
        skmat_tau_real
    )

    for ii in range(skmat_tau_real.shape[0]):

        tau_real = skmat_tau_real[ii, :]

        d_shifted = (
            d_grid
            + displacement[ii, :]
        )

        skmat_tau_redshift[ii, :] = (
            _rsd_delta_kernel_deposit_comoving(
                d_grid,
                tau_real,
                d_shifted,
                dd_cell,
                periodic,
                L_box
            )
        )

    return skmat_tau_redshift

def apply_rsd_old(
    d_grid,
    z_template,
    skmat_tau_real: np.ndarray,
    skmat_vpec: np.ndarray,
    b_doppler: np.ndarray,
    du_cell: float | None = None,
    gamma_lorentz: float = 0.0,
    periodic: bool = True,
    truncate_sigma: float=8.0,
) -> np.ndarray:
    """
    Map a real-space optical-depth skewer into redshift space by
    convolving with a (Voigt/Gaussian) line profile centered on the
    peculiar-velocity-shifted position of each real-space cell:
 
        tau_s(u_i) = sum_j tau_real(u_j) * phi(u_i - u_j - v_pec(u_j); b_j) * du_cell
 
    This is an exact O(N^2) redistribution (optical-depth conserving) that
    allows an arbitrary, per-pixel velocity field and Doppler width --
    appropriate at the "single skewer" level where you don't want the
    approximations needed for an FFT-based (uniform b) convolution. The
    double loop is JIT-compiled and parallelized across the outer
    (redshift-space) index via numba.prange.
 
    Parameters
    ----------
    u_grid : (N,) array
        Line-of-sight coordinate in **velocity units** (km/s), i.e.
        u = a*H*x for comoving distance x. Must be uniformly spaced.
    tau_real : (N,) array
        Real-space optical depth per cell (e.g. from `fgpa_optical_depth`).
    v_pec : (N,) array
        Line-of-sight peculiar velocity at each real-space cell (km/s),
        positive = receding faster / moving away from observer along +u.
    b_doppler : (N,) array
        Per-cell Doppler b-parameter (km/s), e.g. from `doppler_parameter`.
    du_cell : float, optional
        Cell width in km/s; inferred from u_grid if not given.
    gamma_lorentz : float
        Lorentzian HWHM in km/s for a Voigt profile; 0 (default) reduces
        exactly to a pure Gaussian (thermal-only) profile, the standard
        choice for the Lya forest away from DLAs.
    periodic : bool
        If True (default), treat the skewer as one period of a periodic
        simulation box of length N*du_cell and use the minimum-image
        displacement, so optical depth shifted past one edge re-enters at
        the other -- matching how skewers are drawn from a periodic
        N-body/hydro box and avoiding spurious tau loss at the ends of the
        array. Set False for a genuinely non-periodic skewer (e.g. a real
        quasar sightline), where optical depth shifted beyond the array
        edges is lost, as it physically would be for structure extending
        outside the observed segment.
 
    Returns
    -------
    tau_redshift : (N,) array
        Redshift-space optical depth on the same u_grid.
    """

    #skmat_tau_redshift = np.empty_like(skmat_tau_real)

    d_grid = np.asarray(d_grid, dtype=float)
    aa_arr = 1/(1.+z_template)
    HH_arr = H0 * np.sqrt(Om*(1.+z_template)**3 + Ol)
    u_grid = d_grid * (aa_arr*HH_arr)
    u_grid = np.ascontiguousarray(u_grid, dtype=np.float64)
    skmat_tau_real = np.ascontiguousarray(skmat_tau_real, dtype=np.float64)
    skmat_tau_redshift = np.empty_like(skmat_tau_real)
    skmat_vpec = np.ascontiguousarray(skmat_vpec, dtype=np.float64)
    b_doppler = np.ascontiguousarray(b_doppler, dtype=np.float64)
 
    N = u_grid.size
    if du_cell is None:
        du_cell = float(u_grid[1] - u_grid[0])
    else:
        pass
    
    L_box = N * du_cell

    for ii in range(skmat_tau_real.shape[0]):
        tau_real = skmat_tau_real[ii,:]
        vpec = skmat_vpec[ii,:]
 
        u_shifted = u_grid + vpec  # shifted center of each real-space cell's contribution

        if gamma_lorentz == 0.0 and N > 0 and float(np.max(b_doppler)) == 0.0:
            skmat_tau_redshift[ii,:] = _rsd_delta_kernel_deposit(
            u_grid, tau_real, u_shifted, float(du_cell), bool(periodic), float(L_box)
        )

        else:
            if truncate_sigma is not None:
                W = float(truncate_sigma) * float(np.max(b_doppler))
                vmax = float(np.max(np.abs(vpec))) if N > 0 else 0.0
                half_width_pixels = int(math.ceil((W + vmax) / du_cell))

                if 2 * half_width_pixels + 1 < N:
                    skmat_tau_redshift[ii,:] = _rsd_convolution_windowed_loop(
                    u_grid, tau_real, u_shifted, b_doppler,
                    float(du_cell), float(gamma_lorentz), bool(periodic), float(L_box),
                        half_width_pixels,)
                    
            else:
                skmat_tau_redshift[ii,:] = _rsd_convolution_loop(
                u_grid, tau_real, u_shifted, b_doppler,
                float(du_cell), float(gamma_lorentz), bool(periodic), float(L_box),
                    )
        
    return skmat_tau_redshift


@njit(cache=True, fast_math=True, parallel=True)
def apply_rsd_nonumba(d_grid, z_template, skmat_tau_real, v_pec, b_doppler, du_cell, gamma_lorentz, periodic):
    
    """
    Map a real-space optical-depth skewer into redshift space by
    convolving with a (Voigt/Gaussian) line profile centered on the
    peculiar-velocity-shifted position of each real-space cell:
 
        tau_s(u_i) = sum_j tau_real(u_j) * phi(u_i - u_j - v_pec(u_j); b_j) * du_cell
 
    This is an exact O(N^2) redistribution (mass/optical-depth conserving)
    that allows an arbitrary, per-pixel velocity field and Doppler width
    -- appropriate at the "single skewer" level where you don't want the
    approximations needed for an FFT-based (uniform b) convolution.
 
    Parameters
    ----------
    u_grid : (N,) array
        Line-of-sight coordinate in **velocity units** (km/s), i.e.
        u = a*H*x for comoving distance x. Must be uniformly spaced.
    tau_real : (N,) array
        Real-space optical depth per cell (e.g. from `fgpa_optical_depth`).
    v_pec : (N,) array
        Line-of-sight peculiar velocity at each real-space cell (km/s),
        positive = receding faster / moving away from observer along +u.
    b_doppler : (N,) array
        Per-cell Doppler b-parameter (km/s), e.g. from `doppler_parameter`.
    du_cell : float, optional
        Cell width in km/s; inferred from u_grid if not given.
    gamma_lorentz : float
        Lorentzian HWHM in km/s for a Voigt profile; 0 (default) reduces
        exactly to a pure Gaussian (thermal-only) profile, the standard
        choice for the Lya forest away from DLAs.
    periodic : bool
        If True (default), treat the skewer as one period of a periodic
        simulation box of length N*du_cell and use the minimum-image
        displacement, so optical depth shifted past one edge re-enters at
        the other -- matching how skewers are drawn from a periodic
        N-body/hydro box and avoiding spurious tau loss at the ends of the
        array. Set False for a skewer that is genuinely non-periodic
        (e.g. a real quasar sightline), in which case optical depth
        shifted beyond the array edges is lost, as it physically would be
        if the traced structure extends outside the observed segment.
 
    Returns
    -------
    tau_redshift : (N,) array
        Redshift-space optical depth on the same u_grid.
    """
    d_grid = np.asarray(d_grid, dtype=float)
    skmat_tau_real = np.asarray(skmat_tau_real, dtype=float)
    v_pec = np.asarray(v_pec, dtype=float)
    b_doppler = np.asarray(b_doppler, dtype=float)
    
    aa_arr = 1/(1.+z_template)
    HH_arr = H0 * np.sqrt(Om*(1.+z_template)**3 + Ol)
    u_grid = d_grid * (aa_arr*HH_arr)
    
    N = u_grid.size
    if du_cell==None:
        du_cell = float(u_grid[1] - u_grid[0])
 
    # shifted center of each real-space cell's contribution
    u_shifted = u_grid + v_pec  # (N,)
 
    # pairwise displacement: u_i (redshift-space grid) - u_shifted_j (source)
    du = u_grid[:, None] - u_shifted[None, :]  # (N_i, N_j)
 
    if periodic==True:
        L_box = N * du_cell
        du = du - L_box * np.round(du / L_box)  # minimum-image convention
 
    if gamma_lorentz > 0:
        phi = voigt_profile(du, b_doppler[None, :], gamma_lorentz)
    else:
        phi = gaussian_profile(du, b_doppler[None, :])
 
    skmat_tau_redshift = (phi * skmat_tau_real[None, :]).sum(axis=1) * du_cell
    
    return skmat_tau_redshift
 
# **********************************************    
@njit(parallel=False, cache=True, fastmath=True)
def shift_metal(lammin, lammax, ztemplate, lam_rf_lya, lam_rf, skmat, indmat):

    lam_template = (1+ztemplate)*lam_rf_lya
    lam_template = 0.5*(lam_template[1:]+lam_template[:-1])

    lam_arr = (1+ztemplate)*lam_rf

    skmat_new = np.empty_like(skmat)

    for ii in prange(skmat.shape[0]):
        for jj in range(skmat.shape[1]):

            if indmat[ii,jj]>=0:
                if lam_arr[jj]>lammin and lam_arr[jj]<lammax:
                    indtmp = np.argmin(abs(lam_template-lam_arr[jj]))
                    skmat_new[ii,indtmp] +=  skmat[ii,jj]

    return skmat_new

# **********************************************    
@njit(parallel=False, cache=True, fastmath=True)
def regrid_skewers(lammin, lammax, dlam, ztemplate, lam_rf, skmat, indmat):

    numbin = int((lammax-lammin)/dlam)
    lam_template = np.linspace(lammin, lammax, num=numbin+1)
    lam_template_cen = 0.5*(lam_template[1:]+lam_template[:-1])

    lam_arr = (1+ztemplate)*lam_rf

    nobj = skmat.shape[0]
    numbinold = skmat.shape[1]

    skmat_new = np.zeros((nobj, numbin))

    for ii in prange(nobj):
        for jj in range(numbinold):

            if indmat[ii,jj]>=0:
                if lam_arr[jj]>lammin and lam_arr[jj]<lammax:
                    indtmp = int((lam_arr[jj] - lammin)/dlam)
                    skmat_new[ii,indtmp] +=  skmat[ii,jj]

    return skmat_new, lam_template_cen

# ********************************************************
@njit(parallel=False, cache=True,fastmath=True)
def normalize_grf(mat):

    for ii in range(mat.shape[0]):
        mat[ii,:] = mat[ii,:]/np.std(mat[ii,:])

    return mat

# ***************************************************************                                                                       
def convolve_p1d(mat, ztemplate):

    matgauss = np.random.normal(0.,1.,size=(mat.shape[0],mat.shape[1]))
    matgauss_new = 0.*matgauss.copy()

    kk = 2 * np.pi * np.fft.rfftfreq(mat.shape[1], hrbinw)#*100. # In velocity units                                                  
    pk = 1./(1+(kk/k1)**nn) #* kk**0.5
    kern = np.sqrt(pk)

    for ii in range(mat.shape[0]):
        
        fmatgauss = np.fft.rfft(matgauss[ii,:])
        
        fmatgauss *= kern
        fmatgauss[0] = 0.
        matgauss_new[ii,:] = np.fft.irfft(fmatgauss).real

    matgauss_new = normalize_grf(matgauss_new)

    return matgauss_new

def convolve_p1d_ai(mat):

    ngauss = np.random.normal(
        0.0, 1.0,
        size=mat.shape
    )

    matgauss = np.empty_like(ngauss)

    kk = 2.0 * np.pi * np.fft.rfftfreq(mat.shape[1],
        d=PIXEL_WIDTH   # check units!
    )

    pk = 1.0 / (1.0 + (kk/k1)**nn)
    kern = np.sqrt(pk)

    for ii in range(mat.shape[0]):

        fk = np.fft.rfft(ngauss[ii])

        fk *= kern

        # Remove DC mode
        fk[0] = 0.0

        matgauss[ii] = np.fft.irfft(fk,n=mat.shape[1])

    matgauss = normalize_grf(matgauss)

    return matgauss

# **********************************************
@njit(parallel=True, cache=True, fastmath=True)
def add_small_scale_power(ngrid, lbox, skmat, skmat_gauss, indmat, ztemplate, dtemplate):
    
    skmatnew = np.zeros_like(skmat)

    g0 = (5./2.) * Om	/ (Om**(4./7.) - Ol + (1. + Om/2.) * (1. + Ol/70.))
    
    for ii in prange(skmat.shape[0]):
        for jj in range(skmat.shape[1]):  

            if indmat[ii,jj]>=0:                                                                                                                          

                DD = growth_factor(g0, ztemplate[jj])
                
                sigma_gauss = np.exp(A0_1 + A1_1*np.log((1+ztemplate[jj])/(1+z0)))                                                                                    
                rand = normrand * np.exp(DD*sigma_gauss*skmat_gauss[ii,jj] - 0.5*DD**2*sigma_gauss**2) - 1.                                                                               
                                                                                                                
                tmp = skmat[ii,jj]                                                                                                                                 
                ttmp = tmp + rand + tmp*rand #+ (tmp*rand)**2

                aa = np.exp(A0_2 + A1_2*np.log((1+ztemplate[jj])/(1+z0)))
                skmatnew[ii,jj] = aa*(1.+ttmp)**alpha_fgpa

    return skmatnew


# **********************************************
@njit(parallel=False, cache=True, fastmath=True)
def growth_factor(g0, zbin):

    Ez = calcEz(zbin)
    Om_z = calcOm_z(zbin, Ez)
    Ol_z = calcOl_z(zbin, Ez)

    gz = (5./2.) * Om_z / (Om_z**(4./7.) - Ol_z + (1. + Om_z/2.) * (1. + Ol_z/70.))
    D = gz/g0 /(1+zbin)
    
    return D

# **********************************************
@njit(parallel=False, cache=True, fastmath=True)
def calcEz(zbin):

    Hz = np.sqrt(Om*(1.+zbin)**3 + Ol)
    Ez = Hz#/H0

    return Ez

@njit(parallel=False, cache=True, fastmath=True)
def calcOm_z(zbin,Ez):

    Om_z = Om*(1+zbin)**3 / Ez**2

    return Om_z

@njit(parallel=False, cache=True, fastmath=True)
def calcOl_z(zbin,Ez):

    Ol_z = Ol / Ez**2

    return Ol_z
                

# **********************************************
def ensure_regularity(skmat):

    for ii in range(skmat.shape[0]):
        for jj in range(skmat.shape[1]):

            if np.isnan(skmat[ii,jj])==True:
                skmat[ii,jj] = 1.
            elif np.isinf(skmat[ii,jj])==True:
                skmat[ii,jj] = 1.
            elif skmat[ii,jj]>1.:
                skmat[ii,jj] = 1.
            elif skmat[ii,jj]<0:
                skmat[ii,jj]<0.
            else:
                pass
    return skmat

# **********************************************
# **********************************************
# **********************************************
print('--------------------------------')
print('Extract and regrid Lya skewers')
print('--------------------------------')

ti = time.time()

lcell = lbox/ngrid

xobs = obspos[0]
yobs = obspos[1]
zobs = obspos[2]

# Read the tabulated redshift and comoving distance arrays                                                                                                      
zarr = np.fromfile(zarr_filename, dtype=np.float32)
darr = np.fromfile(darr_filename, dtype=np.float32)


print('Read QSO positions ...')
# Now read QSO positions in redshift space                                        
# The containers are used first for RA,DEC,z     
cx = np.fromfile(open(posx_qso_filename, 'r'), dtype=np.float32)
cy = np.fromfile(open(posy_qso_filename, 'r'), dtype=np.float32)
cz = np.fromfile(open(posz_qso_filename, 'r'), dtype=np.float32)
print('... done!')
print('') 

# Now open the fits catalog                                                                                                                                     
rawfits = fits.open(fits_filename)
catfits = rawfits[1].data
ra = catfits['RA']
dec = catfits['DEC']
zz = catfits['Z']
mockid = catfits['MOCKID']

# Read flux field
print('Read flux field ...')

delta = h5py.File(delta_filename, 'r')['value'][()]
delta = np.reshape(delta, (ngrid,ngrid,ngrid))

velx = h5py.File(velx_filename, 'r')['value'][()]
velx = np.reshape(velx, (ngrid,ngrid,ngrid))

vely = h5py.File(vely_filename, 'r')['value'][()]
vely = np.reshape(vely, (ngrid,ngrid,ngrid))

velz = h5py.File(velz_filename, 'r')['value'][()]
velz = np.reshape(velz, (ngrid,ngrid,ngrid))
"""

delta = np.ones((ngrid,ngrid,ngrid))
velx = np.ones((ngrid,ngrid,ngrid))
vely = np.ones((ngrid,ngrid,ngrid))
velz = np.ones((ngrid,ngrid,ngrid))
"""
print('... done!')

# Cut the QSO positions - keep only QSOs relevant for lya                                                                                                       
cx = cx[np.logical_and(zz>zmin, zz<zmax)]
cy = cy[np.logical_and(zz>zmin, zz<zmax)]
cz = cz[np.logical_and(zz>zmin, zz<zmax)]

ra = ra[np.logical_and(zz>zmin, zz<zmax)]
dec = dec[np.logical_and(zz>zmin, zz<zmax)]
mockid = mockid[np.logical_and(zz>zmin, zz<zmax)]
zz = zz[np.logical_and(zz>zmin, zz<zmax)]

#ra_rad = (ra+180.) / 180. * np.pi
#dec_rad = (dec+90.) / 180. * np.pi

healpix = healpy.ang2pix(nside, np.radians(90.-dec), np.radians(ra), nest=True)

# Define the DESI footprint
footprint = fits.open('DESI_footprint_nside16.fits')
dark = footprint[1].data['DESI_DARK']

#print('Extracting skewers ...')

# ****************************************
def extract_and_regrid_parallel(ii):

    #ti = time.time()

    dirnum = int(ii/100)

    if dark[ii]==True:

        print(ii)

        if not os.path.exists(output_dir + '/%d/%d/' %(dirnum,ii)):
            os.mkdir(output_dir + '%d/%d/' %(dirnum,ii))

        xpos = cx[healpix==ii]
        ypos = cy[healpix==ii]
        zpos = cz[healpix==ii]
        mockidd = mockid[healpix==ii]

        
        ztemplate, dtemplate, skmat, skmat_vel, indmat = extract_skewers(xpos, ypos, zpos, zmin, zmax, zarr, darr, hrbinw, delta, velx, vely, velz, ngrid, lbox, xobs, yobs, zobs)

        # ADD SMALL_SCALE FLUCTUATIONS
        # First sample the small scales
        skmat_gauss = convolve_p1d(skmat, ztemplate)

        # Then add them to the skewers
        skmat = add_small_scale_power(ngrid, lbox, skmat, skmat_gauss, indmat, ztemplate, dtemplate)

        # Do metals skewers
        """
        skmat_lyb = Ax_lyb * shift_metal(lammin, lammax, ztemplate, lam_lya, lam_lyb, skmat, indmat) 
        skmat_SiII1260 = Ax_SiII_1260 * shift_metal(lammin, lammax, ztemplate, lam_lya, lam_SiII_1260, skmat, indmat) 
        skmat_SiIII1207 = Ax_SiIII_1207 * shift_metal(lammin, lammax, ztemplate, lam_lya, lam_SiIII_1207, skmat, indmat) 
        skmat_SiII1193 = Ax_SiII_1193 * shift_metal(lammin, lammax, ztemplate, lam_lya, lam_SiII_1193, skmat, indmat) 
        skmat_SiII1190 = Ax_SiII_1190 * shift_metal(lammin, lammax, ztemplate, lam_lya, lam_SiII_1190, skmat, indmat) 
        skmat_CIV = Ax_CIV * shift_metal(lammin, lammax, ztemplate, lam_lya, lam_CIV, skmat, indmat) 
        skmat_metals = skmat_lyb + skmat_SiII1260 + skmat_SiIII1207 + skmat_SiII1193 + skmat_SiII1190 + skmat_CIV 

        skmat_lyb += skmat_lya
        skmat_SiII1260 += skmat_lya
        skmat_SiIII1207 += skmat_lya
        skmat_SiII1193 += skmat_lya
        skmat_SiII1190 += skmat_lya
        skmat_CIV += skmat_lya
        skmat_metals += skmat_lya
        """
        
        # Redshift space distortions
        b_doppler = np.zeros(len(dtemplate))
        skmat_lya = apply_rsd(dtemplate, ztemplate, skmat, skmat_vel, b_doppler,periodic=False)

        """
        skmat_lyb = apply_rsd(dtemplate, ztemplate, skmat_lyb, skmat_vel, b_doppler)
        skmat_SiII1260 = apply_rsd(dtemplate, ztemplate, skmat_SiII1260, skmat_vel, b_doppler)
        skmat_SiIII1207 = apply_rsd(dtemplate, ztemplate, skmat_SiIII1207, skmat_vel, b_doppler)
        skmat_SiII1193 = apply_rsd(dtemplate, ztemplate, skmat_SiII1193, skmat_vel, b_doppler)
        skmat_SiII1190 = apply_rsd(dtemplate, ztemplate, skmat_SiII1190, skmat_vel, b_doppler)
        skmat_CIV = apply_rsd(dtemplate, ztemplate, skmat_CIV, skmat_vel, b_doppler)
        skmat_metals = apply_rsd(dtemplate, ztemplate, skmat_metals, skmat_vel, b_doppler)
        """

        # REGRIDDING
        skmat_lya, lam_template = regrid_skewers(lammin, lammax, dlam, ztemplate, lam_lya, skmat_lya, indmat)

        """
        skmat_lyb, lam_template = regrid_skewers(lammin, lammax, dlam, ztemplate, lam_lya, skmat_lyb, indmat)
        skmat_SiII1260, lam_template = regrid_skewers(lammin, lammax, dlam, ztemplate, lam_lya, skmat_SiII1260, indmat)
        skmat_SiIII1207, lam_template = regrid_skewers(lammin, lammax, dlam, ztemplate, lam_lya, skmat_SiIII1207, indmat)
        skmat_SiII1193, lam_template = regrid_skewers(lammin, lammax, dlam, ztemplate, lam_lya, skmat_SiII1193, indmat)
        skmat_SiII1190, lam_template = regrid_skewers(lammin, lammax, dlam, ztemplate, lam_lya, skmat_SiII1190, indmat)
        skmat_CIV, lam_template = regrid_skewers(lammin, lammax, dlam, ztemplate, lam_lya, skmat_CIV, indmat)
        skmat_metals, lam_template = regrid_skewers(lammin, lammax, dlam, ztemplate, lam_lya, skmat_metals, indmat) 
        """

        # PASS FROM TAU TO FLUX
        skmat_lya = tau_to_flux(ngrid, lbox, skmat_lya)
        """
        skmat_lyb = tau_to_flux(ngrid, lbox, skmat_lyb)
        skmat_SiII_1260 = tau_to_flux(ngrid, lbox, skmat_SiII_1260)
        skmat_SiIII_1207 = tau_to_flux(ngrid, lbox, skmat_SiIII_1207)
        skmat_SiII_1193 = tau_to_flux(ngrid, lbox, skmat_SiII_1193)
        skmat_SiII_1190 = tau_to_flux(ngrid, lbox, skmat_SiII_1190)
        skmat_CIV = tau_to_flux(ngrid, lbox, skmat_CIV)
        skmat_metals = tau_to_flux(ngrid, lbox, skmat_metals)
        """

        # Flux diagnostics
        #skmat_lya = ensure_regularity(skmat_lya)
        print('Min, max, mean flux lya: ', np.amin(skmat_lya), np.amax(skmat_lya), np.mean(skmat_lya))
        
        # Now construct the fits file
        # Read QSO catalog and build HDU
        rawqso = fits.open(output_dir + 'master.fits')

        qsotab = rawqso[1].data
        arr, inddx, inddx2 = np.intersect1d(qsotab['MOCKID'], mockidd, return_indices=True)
        data = qsotab[inddx]
        #data2 = qsotab[inddx2]

        #print(zztmp/data['Z'])
        #print(zztmp/data2['Z'])
        
        c1 = fits.Column(name='RA', array=data['RA'], format='D')
        c2 = fits.Column(name='DEC', array=data['DEC'], format='D')
        c3 = fits.Column(name='Z_noRSD', array=data['Z_noRSD'], format='D')
        c4 = fits.Column(name='Z', array=data['Z'], format='D')
        c5 = fits.Column(name='MOCKID', array=data['MOCKID'], format='D')
        hdu1 = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5], name='METADATA')

        # Read DLA catalog and build HDU
        rawdla = fits.open(output_dir + 'master_DLA.fits')

        dlatab = rawdla[1].data
        arr, inddx, inddx2 = np.intersect1d(dlatab['MOCKID'], mockidd, return_indices=True)
        data = dlatab[inddx]

        c1 = fits.Column(name='RA', array=data['RA'], format='D')
        c2 = fits.Column(name='DEC', array=data['DEC'], format='D')
        c3 = fits.Column(name='Z_DLA_NO_RSD', array=data['Z_DLA_NO_RSD'], format='D')
        c4 = fits.Column(name='Z_DLA_RSD', array=data['Z_DLA_RSD'], format='D')
        c5 = fits.Column(name='MOCKID', array=data['MOCKID'], format='D')
        c6 = fits.Column(name='DLAID', array=data['DLAID'], format='D')
        c7 = fits.Column(name='N_HI_DLA', array=data['NHI_DLA'], format='D')
        hdudla = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7], name='DLA')

        hdu_list = fits.HDUList([
            fits.PrimaryHDU(),
            hdu1,
            fits.ImageHDU(lam_template),
            fits.ImageHDU(skmat_lya),
            #fits.ImageHDU(skmat_lyb),
            #fits.ImageHDU(skmat_metals),
            hdudla])

        hdu_list[1].name = 'METADATA'
        hdu_list[2].name = 'WAVELENGTH'
        hdu_list[3].name = 'F_LYA'
        """
        hdu_list[4].name = 'F_LYA_LYB'
        hdu_list[5].name = 'F_LYA_METALS'
        hdu_list[5].name = 'F_LYA_SiII1260'
        hdu_list[5].name = 'F_LYA_SiIII1207'
        hdu_list[5].name = 'F_LYA_SiII1193'
        hdu_list[5].name = 'F_LYA_SiII1190'
        hdu_list[5].name = 'F_LYA_CIV'
        """
        #hdu_list[6].name = 'DLA'

        hdu_list[1].header['HPXNSIDE'] = 16
        hdu_list[1].header['HPXPIXEL'] = ii
        hdu_list[1].header['HPXNEST'] = True
        hdu_list[1].header['LYA'] = 1215.67

        hdu_list.writeto(output_dir + '%d/%d/transmission-%d-%d.fits.gz' %(dirnum,ii,nside,ii), overwrite=True)

        #print(ii, (tf0-ti0)/60, (tf1-ti1)/60., (tf2-ti2)/60., (tf3-ti3)/60., (tf4-ti4)/60., (tf5-ti5)/60.)

        #break
        

# ******************************************************************

tin = time.time()

pixx = np.array([1200])

# First check if the master directory exists. If not, create it, together with the subdirectories
if not os.path.exists('/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/%s/' %version):
    os.mkdir('/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/%s/' %version)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

#for ii in pixx:
for ii in range(pixmax):
    if not os.path.exists(output_dir + '%d/' %(ii//100)):
        os.mkdir(output_dir + '%d/' %(ii//100))

# Now start extraction and regridding
#ii_list = [(ii) for ii in pixx]
ii_list = [(ii) for ii in range(pixmax)]    

if num_processes<0:
    num_processes = cpu_count()
else:
    num_processes = num_processes

with Pool(processes=num_processes) as pool:
    pool.map(extract_and_regrid_parallel, ii_list)

tfin = time.time()

dt = (tfin-tin)/60.

print('Elapsed ' + str(dt) + ' minutes ...')
