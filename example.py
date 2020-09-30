#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
Version: September 2020
'''

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

import FDCA

data_path = '/Users/jortboxelaar/Documents/Research/Halo-FDCA/Data/A2744_JVLA.image.fits'
#data_path = '/Users/jortboxelaar/Desktop/PSZ2G086.93+53.18_cut2.fits'
mask_path = '/Users/jortboxelaar/Documents/Research/Halo-FDCA/Data/Masks/A2744halo.reg'
#mask_path = '--'

def get_initial_guess(halo):
    r_guess = halo.radius/(3.5*halo.pix_size)
    r_bound = halo.data.shape[0]/2.
    if r_guess >= halo.data.shape[1]/2.: r_guess = halo.data.shape[1]/4.

    diff   = np.abs(halo.margin)
    p0     = (np.max(halo.data.value), halo.centre_pix[0]+diff[0],
              halo.centre_pix[1]+diff[2], r_guess,r_guess,r_guess,r_guess,0.,0.,0.)
    bounds = ([0.,0.,0.,0.,0.,0.,0.,-np.inf, 0., -np.inf],
              [np.inf,halo.data.shape[0],halo.data.shape[1],
               r_bound,r_bound,r_bound,r_bound,np.inf, np.inf, np.inf])
    return p0,bounds

#loc  = SkyCoord('15 13 54.8967 +52 47 54.300', unit=(u.hourangle, u.deg), frame='icrs')
#halo = FDCA.Radio_Halo('PSZ2G086.93+53.18', data_path, z=0.6752, loc=loc, decreased_fov=True, M500=None, R500=None, spectr_index=-1.2)

loc  = SkyCoord('00 14 20.03 -30 23 17.8', unit=(u.hourangle, u.deg), frame='icrs')
halo = FDCA.Radio_Halo('Abell2744', data_path, z=0.308, loc=loc, decreased_fov=True, M500=None, R500=None, spectr_index=-1.2)
p0, bounds = get_initial_guess(halo)

'''
fit  = FDCA.markov_chain_monte_carlo.fitting(halo, halo.data_mcmc, 'circle', p0,
                                             bounds, walkers=10, steps=20,
                                             burntime=None, logger=halo.log, mask=True,
                                             maskpath=mask_path, max_radius=None,
                                             gamma_prior=False, k_exponent=False,)
fit.__preFit__()
fit.__run__(save=True)
'''

processing = FDCA.markov_chain_monte_carlo.processing(halo, halo.data, 'circle',
                                                    logger=halo.log,mask=True,
                                                    maskpath=mask_path, save=True,
                                                    k_exponent=False, offset=False,
                                                    burntime=None)
processing.plot_results()
processing.cornerplot()
processing.get_chi2_value()
processing.get_flux(int_max=np.inf, freq=None)
processing.get_power(freq=None)
[conf_low, conf_up] = processing.get_confidence_interval(percentage=95, units=True)

halo.Close()
