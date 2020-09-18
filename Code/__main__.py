#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
Version: 08 June 2020
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
import make_all_info_file as make_all_info_file

Jydeg2     = u.Jy/(u.deg*u.deg)
mJyarcsec2 = u.mJy/(u.arcsec*u.arcsec)
uJyarcsec2 = 1.e-3*u.mJy/(u.arcsec*u.arcsec)

def FindTargets():
    targets  = list()
    path     = os.getcwd()
    for i in reversed(range(len(path))):
        if path[i-9:i] == 'Halo-FDCA':
            basedir = path[:i]+'/'

    data_file = basedir+'Data/database.dat'
    targets = np.genfromtxt(data_file, delimiter=',', dtype='U160')
    if targets.shape == (3,):
        targets = targets.reshape((1,3))
    return targets,basedir

def MCMC_run(halo, maskpath, logger, dim='rotated_ellipse'):
    r_guess = halo.radius/(3.5*halo.pix_size)
    r_bound = halo.data.shape[0]/2.
    if r_guess >= halo.data.shape[1]/2.:
        r_guess = halo.data.shape[1]/4.

    diff   = np.abs(halo.margin)
    p0     = (np.max(halo.data.value), halo.centre_pix[0]+diff[0],
                halo.centre_pix[1]+diff[2], r_guess,r_guess,r_guess,r_guess,0.,0.,0.)
    bounds = ([0.,0.,0.,0.,0.,0.,0.,-np.inf, 0., -np.inf],
              [np.inf,halo.data.shape[0],halo.data.shape[1],
                    r_bound,r_bound,r_bound,r_bound,np.inf, np.inf, np.inf])

    if dim not in ['circle','ellipse', 'rotated_ellipse', 'skewed']:
        print('Provide valid function kind')
        sys.exit()

    fit = mcmc.fitting(halo, halo.data_mcmc, dim, p0, bounds,
                        walkers=200, steps=1200, save=True, mask=True, burntime=250,
                        logger=halo.log, maskpath=maskpath, max_radius=None, k_exponent=False)
    fit.__preFit__()
    fit.__run__()

def MCMC_retreival(halo, maskpath, logger, dim='circle'):
    processing = mcmc.processing(halo, halo.data, dim=dim,logger=logger,
                                    mask=True, maskpath=maskpath, k_exponent=False)
    processing.plot_results()
    #print(halo.original_image.shape)
    #print(halo.pix_size)
    #print(halo.bmaj, halo.bmin, halo.bpa)
    processing.get_chi2_value()
    processing.get_flux(int_max=10)
    #processing.get_power(freq=150*u.MHz)

    #processing.tableprint()
    #print(processing.power_val.value)
    #print(processing.params_units[3])
    return processing

def worker_process(object, path, maskpath, logger):
    # Model to use for fitting
    if object == 'Abell2744':
        loc = SkyCoord('00 14 20.03 -30 23 17.8', unit=(u.hourangle, u.deg), frame='icrs')
        halo = HaloObject.Radio_Halo(object, path, logger=logger, decreased_fov=True, z=0.308, loc=loc)
    elif object == 'RXCJ1825.3+3026':
        loc = SkyCoord('18 25 20.0 +30 26 11.2', unit=(u.hourangle, u.deg))
        halo = HaloObject.Radio_Halo(object, path, logger=logger, decreased_fov=True, z=0.065, loc=loc)
    elif object == 'Phoenix':
        loc = SkyCoord('23 44 43.9 -42 43 13', unit=(u.hourangle, u.deg))
        halo = HaloObject.Radio_Halo(object, path, logger=logger, decreased_fov=False, z=0.597, loc=loc)
    else:
        halo = HaloObject.Radio_Halo(object, path, logger=logger, decreased_fov=True)

    MCMC_run(halo, maskpath, logger, dim='circle')
    MCMC_run(halo, maskpath, logger, dim='rotated_ellipse')
    MCMC_run(halo, maskpath, logger, dim='skewed')

    #try:
    halo.result4 = MCMC_retreival(halo, maskpath, logger, dim='circle')
    halo.result6 = MCMC_retreival(halo, maskpath, logger, dim='rotated_ellipse')
    halo.result8 = MCMC_retreival(halo, maskpath, logger, dim='skewed')
    #except:
    #    pass

    #plot.model_comparisson(halo, mask=halo.result4.mask)
    #objectlist = [halo.result4, halo.result6, halo.result8]
    #best_model_num = np.argmin([halo.result4.AICc, halo.result6.AICc, halo.result8.AICc])
    #make_all_info_file.main(halo, objectlist[best_model_num])
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
    halos = list()

    for target in targets:
        logger = logging.getLogger(str(target[0])[4:])
        logger.log(logging.INFO, 'Start Process for: '+ target[1])
        #try:
        halo = worker_process(target[0], target[1], target[2], logger)
        halos.append(halo)
        #except Exception as e:
        #    logger.log(logging.CRITICAL, 'process for '+target[0]+\
        #                        'failed with error message')
        #    print(e)

    #utils.HaloStatistics(halos)
