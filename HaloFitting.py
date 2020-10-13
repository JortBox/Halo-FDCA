#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
Version: October 2020
'''

from astropy.coordinates import SkyCoord
import logging
import os
from datetime import datetime
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
parser.add_argument('-out_path',help='(str) Path to code output. Default: directory code is in.', default='./', type=str)
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

args = vars(parser.parse_args())
#data_path = './Data/A2744_JVLA.image.fits'
#mask_path = './Data/Masks/A2744halo.reg'


def init_logging(args):
    path = args['out_path']
    now = str(datetime.now())[:19]
    if not os.path.exists(path+'log/'):
        os.makedirs(path+'/log/')

    d = {
            'version': 1,
            'formatters': {
            'detailed': {
                'class': 'logging.Formatter',
                'format': '%(asctime)s %(name)-12s %(processName)-2s %(levelname)-8s %(message)s'
            }
            },
            'handlers': {
            'file': {
                'class': 'logging.FileHandler',
                'filename': path+'log/'+args['o']+' '+now+'.log',
                'mode': 'w',
                'formatter': 'detailed',
            },
            },
            'root': {
            'level': 'INFO',
            'handlers': ['file'] #,'console'
            },
        }

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.config.dictConfig(d)
    return logging

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

loc = args['loc']
if args['freq'] != None: args['freq'] *= 1.*u.MHz
if loc is not None:
    loc  = SkyCoord(args['loc'], unit=(u.hourangle, u.deg), frame=args['frame'])

logging = init_logging(args)
logger  = logging.getLogger(args['o'])
logger.log(logging.INFO, 'Start Process for: '+ args['o'])
logger.log(logging.INFO, 'Run Arguments: \n'+ str(args)+ '\n')


halo = FDCA.HaloObject.Radio_Halo(args['o'], args['d_path'], maskpath=args['m_path'], mask=args['m'],
                            decreased_fov=args['fov'],logger=logger, loc=loc,
                            M500=None, R500=None, z=args['z'],
                            outputpath=args['out_path'], spectr_index=args['spectr_idx'])
p0, bounds = get_initial_guess(halo)

'''
fit  = FDCA.markov_chain_monte_carlo.fitting(halo, halo.data_mcmc, args['model'], p0,
                                             bounds, walkers=args['walkers'], steps=args['steps'],
                                             burntime=args['burntime'], logger=halo.log, mask=args['m'],
                                             maskpath=args['m_path'], max_radius=args['max_radius'],
                                             gamma_prior=args['gamma_prior'], k_exponent=args['k_exp'])
fit.__preFit__()
fit.__run__(save=args['s'])
'''

processing = FDCA.markov_chain_monte_carlo.processing(halo, halo.data, args['model'],
                                                    logger=halo.log,mask=args['m'],
                                                    maskpath=args['m_path'], save=args['s'],
                                                    k_exponent=args['k_exp'], offset=False,
                                                    burntime=args['burntime'])
processing.plot_results()
processing.get_chi2_value()
processing.get_flux(int_max=args['int_max'], freq=args['freq'])
processing.get_power(freq=args['freq'])
[conf_low, conf_up] = processing.get_confidence_interval(percentage=95, units=True)
#processing.tableprint()
#halo.result4 = processing

#FDCA.plotting_fits.model_comparisson(halo, mask=halo.result4.mask)

halo.Close()
#python3 HaloFitting.py -o Abell2744 -d_path ./ExampleData/A2744_JVLA.image.fits -m_path ./ExampleData/Masks/A2744halo.reg -loc '00 14 20.03 -30 23 17.8' -z 0.308 -model circle
