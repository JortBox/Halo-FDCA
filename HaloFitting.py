#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
Version: October 2020
'''

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import argparse

import FDCA

parser = argparse.ArgumentParser(description='Halo-FDCA: A automated flux density calculator for radio halos in galaxy clusters. (Boxelaar et al.)')
parser.add_argument('-o',help='(str) Cluster object name', required=True, type=str)
parser.add_argument('-model',help='(str) Model to use. choose from (circle, ellipse, rotated_ellipse, skewed). Default: circle',default='circle', type=str)
parser.add_argument('-frame',help='(str) Coordinate frame. Default: ICRS', default='icrs', type=str)
parser.add_argument('-loc',help="(str) Sky coordinates of cluster. provide coordinates of the form: 'hh mm ss.ss -dd mm ss.s' in hourangle units. Default: None and image centre is chosen.", default = None, type=str)
parser.add_argument('-z',help='(float) cluster redshift', default=None, type=float)
parser.add_argument('-m',help='(bool) choose to include mask or not. If True, -maskPath should be specified. Default: True',default=True, type=bool)
parser.add_argument('-d_path',help='(str) Path to clusters radio FITS image.', required=True, type=str)
parser.add_argument('-m_path',help='(str) Path to clusters .reg mask file.', required=True, type=str)
parser.add_argument('-fov',help='(bool) Declare if image size has to be decreased before MCMC-ing. Amount of decreasement has ben automatically set to 3.5*r_e. Default: True',default=True, type=bool)
parser.add_argument('-spectr_idx',help='(float) Set spectral index of cluster (S ~ nu^alpha). Used to calculate power and extrapolate flux to arbitrary frequencies. Default: -1.2',default=-1.2, type=float)
parser.add_argument('-walkers',help='(int) Number of walkers to deploy in the MCMC algorithm. Default: 200',default=200, type=int)
parser.add_argument('-steps',help='(int) Number of evauations each walker has to do. Default: 1200',default=1200, type=int)
parser.add_argument('-burntime',help='(int) Burn-in time for MCMC walkers. See emcee documentation for info. Default: None',default=None, type=int)
parser.add_argument('-max_radius',help='(float) Maximum posiible radius cut-off. Fitted halos cannot have any r > max_radius. In units of kpc. Default: None (implying image_size/2).',default=None, type=float)
parser.add_argument('-gamma_prior',help='(bool) Wether to use a gamma distribution as a prior for radii. Default is False. For the gamma parameters: shape = 2.5, scale = 120 kpc. Default: False',default=False, type=bool)
parser.add_argument('-k_exp',help='(bool) Wether to use k exponent to change shape of exponential distribution. Default: False',default=False, type=int)
parser.add_argument('-s',help='(bool) Whether to save the mcmc sampler chain in a fits file. When true, k will be included as parameter. Default: True.',default=True, type=bool)
parser.add_argument('-int_max',help='(float) Integration radius in r_e units. Default: inf',default=np.inf, type=float)
parser.add_argument('-freq',help='(float) frequency in MHz to calculate flux in. When given, the spectral index will be used. Default: image frequency',default=None, type=str)
#parser.add_argument('-',help='', type=str)

args = vars(parser.parse_args())

#data_path = './Data/A2744_JVLA.image.fits'
#mask_path = './Data/Masks/A2744halo.reg'

if args['freq'] != None: args['freq'] *= 1.*u.MHz

def get_initial_guess(halo):
    r_guess = halo.radius/(3.5*halo.pix_size)
    r_bound = halo.data.shape[0]/2.
    if r_guess >= halo.data.shape[1]/2.: r_guess = halo.data.shape[1]/4.

    diff   = np.abs(halo.margin)
    p0     = (halo.I0, halo.centre_pix[0]+diff[0],
              halo.centre_pix[1]+diff[2], r_guess,r_guess,r_guess,r_guess,0.,0.,0.)
    bounds = ([0.,0.,0.,0.,0.,0.,0.,-np.inf, 0., -np.inf],
              [np.inf,halo.data.shape[0],halo.data.shape[1],
               r_bound,r_bound,r_bound,r_bound,np.inf, np.inf, np.inf])
    return p0,bounds

#loc  = SkyCoord('15 13 54.8967 +52 47 54.300', unit=(u.hourangle, u.deg), frame='icrs')
#halo = FDCA.Radio_Halo('PSZ2G086.93+53.18', data_path, z=0.6752, loc=loc, decreased_fov=True, M500=None, R500=None, spectr_index=-1.2)

loc = args['loc']
if loc is not None:
    loc  = SkyCoord(args['loc'], unit=(u.hourangle, u.deg), frame=args['frame'])

halo = FDCA.Radio_Halo(args['o'], args['d_path'], z=args['z'], loc=loc, maskpath=args['m_path'], mask=args['m'], decreased_fov=args['fov'], M500=None, R500=None, spectr_index=args['spectr_idx'])
p0, bounds = get_initial_guess(halo)


fit  = FDCA.markov_chain_monte_carlo.fitting(halo, halo.data_mcmc, args['model'], p0,
                                             bounds, walkers=args['walkers'], steps=args['steps'],
                                             burntime=args['burntime'], logger=halo.log, mask=args['m'],
                                             maskpath=args['m_path'], max_radius=args['max_radius'],
                                             gamma_prior=args['gamma_prior'], k_exponent=args['k_exp'])
fit.__preFit__()
fit.__run__(save=args['s'])

processing = FDCA.markov_chain_monte_carlo.processing(halo, halo.data, args['model'],
                                                    logger=halo.log,mask=args['m'],
                                                    maskpath=args['m_path'], save=args['s'],
                                                    k_exponent=args['k_exp'], offset=False,
                                                    burntime=args['burntime'])
processing.plot_results()
processing.cornerplot()
processing.get_chi2_value()
processing.get_flux(int_max=np.inf, freq=args['freq'])
processing.get_power(freq=args['freq'])
[conf_low, conf_up] = processing.get_confidence_interval(percentage=95, units=True)

halo.Close()
