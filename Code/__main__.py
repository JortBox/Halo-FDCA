#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
Version: 21 June 2020
'''
import logging
import sys, os
import logging.config
import logging.handlers
from datetime import datetime
from multiprocessing import Process
from astropy.coordinates import SkyCoord

import astropy.units as u
import numpy as np

import __HaloObject__ as HaloObject
import plotting_fits as plot
import utils
import markov_chain_monte_carlo as mcmc

Jydeg2     = u.Jy/(u.deg*u.deg)
mJyarcsec2 = u.mJy/(u.arcsec*u.arcsec)
uJyarcsec2 = 1.e-3*u.mJy/(u.arcsec*u.arcsec)

def FindTargets():
    path     = os.getcwd()
    basedir  = '/'.join(path.split('/')[:-1])+'/'

    data_file = basedir+'Data/database.dat'
    targets = np.genfromtxt(data_file, delimiter=',', dtype='U160')
    if targets.shape == (3,):
        targets = targets.reshape((1,3))
    return targets,basedir

def MCMC_run(halo, maskpath, logger, dim='rotated_ellipse'):
    r_guess = halo.radius/(3.5*halo.pix_size)
    r_bound = halo.data.shape[0]/2.
    if halo.mini:
        r_guess /= 9.
        r_bound = r_guess*4.
    if r_guess >= halo.data.shape[1]/2.:
        r_guess = halo.data.shape[1]/4.
    diff    = np.abs(halo.margin)

    if dim == 'skewed':
        p0     = (np.max(halo.data.value), halo.centre_pix[0]+diff[0],
                    halo.centre_pix[1]+diff[2], r_guess,r_guess,r_guess,r_guess,0.)
        bounds = ([0.,0.,0.,0.,0.,0.,0.,-np.inf],
              [np.inf,halo.data.shape[0],halo.data.shape[1],
                    r_bound,r_bound,r_bound,r_bound,np.inf])

    elif dim == 'rotated_ellipse':
        p0     = (np.max(halo.data.value), halo.centre_pix[0]+diff[0],
                    halo.centre_pix[1]+diff[2], r_guess,r_guess,0.)
        bounds = ([0.,0.,0.,0.,0.,-np.inf],
                  [np.inf,halo.data.shape[0],halo.data.shape[1],r_bound,r_bound,np.inf])
    elif dim == 'ellipse':
        p0     = (np.max(halo.data.value), halo.centre_pix[0]+diff[0],
                    halo.centre_pix[1]+diff[2], r_guess,r_guess)
        bounds = ([0.,0.,0.,0.,0.],
                  [np.inf,halo.data.shape[0],halo.data.shape[1],r_bound,r_bound])
    elif dim == 'circle':
        p0     = (np.max(halo.data.value), halo.centre_pix[0]+diff[0],
                    halo.centre_pix[1]+diff[2], r_guess)
        bounds = ([0.,0.,0.,0.,],
                  [np.inf,halo.data.shape[0],halo.data.shape[1],r_bound])
    else:
        print('Provide valid function kind')
        sys.exit()

    fit = mcmc.fitting(halo, halo.data_mcmc, dim, p0, bounds,
                            walkers=200, steps=1000, save=True, mask=True,
                            burntime=300, logger=halo.log, rebin=True, maskpath=maskpath)
    fit.__run__()

def MCMC_retreival(halo, maskpath, logger, dim='circle'):
    processing = mcmc.processing(halo, halo.data, dim=dim,logger=logger,
                                    mask=True,rebin=True, maskpath=maskpath)
    processing.plot_results()
    processing.get_flux(freq=144*u.MHz)
    processing.get_power(freq=150*u.MHz)
    processing.get_chi2_value()
    return processing

def worker_process(object, path, maskpath, logger):
    model = 'circle' #can be circle, rotated_ellipse or skewed
    if object=='Abell1033':
        halo = HaloObject.Radio_Halo(object, path, logger=logger, decreased_fov=True, z=0.122)

    MCMC_run(halo, maskpath, logger, dim=model)
    try:
        halo.result = MCMC_retreival(halo, maskpath, logger, dim=model)
    except Exception as e:
        print(e)
        print('MCMC retrieval Failed')
        pass

    halo.Close()
    return halo

if __name__ == '__main__':
    CurrTime = datetime.now()
    now = str(CurrTime)[:19]
    targets, basedir = FindTargets()
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
                'filename': basedir+'Code/.log/'+now+'.log',
                'mode': 'w',
                'formatter': 'detailed',
            },
            },
            'root': {
            'level': 'INFO',
            'handlers': ['file'] #,'console'
            },
        }
    if not os.path.exists(basedir+'Code/.log/'):
        os.makedirs(basedir+'Code/.log/')

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.config.dictConfig(d)
    mass = list()
    power  = list()

    for target in targets:
        logger = logging.getLogger(str(target[0])[4:])
        logger.log(logging.INFO, 'Start Process for: '+ target[1])
        try:
            halo = worker_process(target[0], target[1], target[2], logger)
        except Exception as e:
            logger.log(logging.CRITICAL, 'process for '+target[0]+\
                                'failed with error message: See terminal')
            print(e)
            pass
