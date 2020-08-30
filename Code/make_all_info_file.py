#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
Version: 05 May 2020
'''
# Built in module imports
import sys
import os
import logging
import time
from multiprocessing import Pool

# Scipy, astropy, emcee imports
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
import astropy.units as u
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import emcee
import corner
import pandas as pd

# Subfiles imports
import plotting_fits as plot
import markov_chain_monte_carlo
import utils

plt.rc('text',usetex=True)
plt.rc('font', family='serif')

rad2deg    = 180./np.pi
deg2rad    = np.pi/180.
Jydeg2     = u.Jy/(u.deg*u.deg)
mJyarcsec2 = u.mJy/(u.arcsec*u.arcsec)
uJyarcsec2 = 1.e-3*u.mJy/(u.arcsec*u.arcsec)


targets  = list()
path     = os.getcwd()
for i in reversed(range(len(path))):
    if path[i-24:i] == 'RadioHalo_FluxCalculator':
        basedir = path[:i]+'/'


def FindTargets():
    file_name =  basedir+'Data/table_results.xlsx'# path to file + file name
    sheet =  ['Sheet 1', 'Sheet 2', 'Sheet 3']# sheet name or sheet number or list of sheet numbers and names
    df = pd.read_excel(io=file_name, sheet_name=sheet)
    #print(df['Sheet 1'][index[0]])
    targets = df['Sheet 1']['name']
    flux4 = df['Sheet 1']['4_param']
    flux6 = df['Sheet 1']['6_param']
    flux8 = df['Sheet 1']['8_param']
    fluxes = np.array([flux4,flux6,flux8])
    chi4 = df['Sheet 3']['4_param']
    chi6 = df['Sheet 3']['6_param']
    chi8 = df['Sheet 3']['8_param']
    chi2 = np.array([chi4,chi6,chi8])
    for i in range(fluxes.shape[0]):
        for j in range(fluxes.shape[1]):
            fluxes[i,j]=fluxes[i,j].replace(' mJy', '')
            fluxes[i,j]=fluxes[i,j].replace(' +/- ', ',')
    fluxes = fluxes[0,:]

    return targets, fluxes

def estimate_mcxc_mass_err(true):
    slope, slopeerr = 1.19 , 0.10 # Based on fitting with MCXC/PSZ2 overlap
    mass     = true * np.random.normal(slope, slopeerr)
    mass_err = 0.33 #Average error on PSZ2 mass

    a         = np.random.normal(slope, 2.8*slopeerr, 500)
    mass_dist = np.random.normal(mass, mass_err, 500)
    mass_mcxc = mass/a
    return np.std(mass_mcxc)

def extract_object_info(target):
    '''Written for MCXC catalogue. Information is gathered from there. If custom
    parameters are given, these will be used. if nothing is found, filling
    values are set. This is only a problem if you try to calculate radio power.'''

    if target[:4] == 'MCXC':
        cat = 'J/A+A/534/A109/mcxc'
    elif target[:4] == 'PSZ2':
        cat = 'J/A+A/594/A27/psz2'
    else:
        cat=None
    table     = Vizier.query_object(target,catalog=cat)

    if target[:4] == 'MCXC':
        M500 = float(table[cat]['M500'][0])
        M500_err_up = estimate_mcxc_mass_err(M500)
        M500_err_down = estimate_mcxc_mass_err(M500)
        z    = float(table[cat]['z'][0])
        R500 = float(table[cat]['R500'][0])
        
        coord    = str(table[cat]['RAJ2000'][0])+' '+ str(table[cat]['DEJ2000'][0])
        loc = SkyCoord(coord, unit=(u.hourangle,u.deg))
        RA, DEC = loc.ra.deg, loc.dec.deg

    elif target[:4] == 'PSZ2':
        M500 = float(table[cat]['MSZ'][0])
        M500_err_up = float(table[cat]['E_MSZ'][0])
        M500_err_down = float(table[cat]['e_MSZ'][0])
        z    = float(table[cat]['z'][0])
        R500 = None
        
        coord = [table[cat]['RAJ2000'][0],table[cat]['DEJ2000'][0]]
        loc = SkyCoord(coord[0], coord[1], unit=u.deg)
        RA, DEC = loc.ra.deg, loc.dec.deg
    else:
        print('characteristic error: ', target)
        M500 = 0.
        R500 = None
        M500_err_up = 0.
        M500_err_down = 0.
        z    = 0.

    return M500, M500_err_up, M500_err_down, z, R500, RA, DEC

def main(halo, result):
    try:
        M500, M500_err_up, M500_err_down, z, R500, RA, DEC = extract_object_info(halo.target)
        flux_value = result.flux_val.value*1.e-3
        flux_err = result.flux_std.value*1.e-3
        #file_arr.append(np.array([halo.target, z, flux_value, flux_err, 144., 0.002, M500, M500_err_up, M500_err_down, 0.,0.]))
        print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(halo.target, z, flux_value, flux_err, 144., 0.002, M500, M500_err_up, M500_err_down, 0.,0.))
        #if halo.name[:4]=='MCXC':
        #    print('\\text{%s} & %.2f & %.2f & %.2f & %.3f & %.2f \\\\' % (halo.name, RA, DEC, M500, z, R500))
        #else:
        #    print('\\text{%s} & %.2f & %.2f & %.1f^{%.1f}_{%.1f} & %.3f &  \\\\' % (halo.name, RA, DEC, M500, M500_err_up, M500_err_down, z))
    except:
        print('FAIL')
