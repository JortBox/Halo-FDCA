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
from . import HaloObject
from . import markov_chain_monte_carlo
from . import fdca_utils as utils
from . import plotting_fits

def main():
    dir = os.getcwd()
    now = str(datetime.now())[:19]
    if not os.path.exists(dir+'/log/'):
        os.makedirs(dir+'/log/')
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
                'filename': dir+'/log/'+now+'.log',
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

def Radio_Halo(object, path, decreased_fov=True, maskpath=None, mask=False, loc=None, M500=None, R500=None,
                z=None, spectr_index=-1.2):
    logging = main()
    logger  = logging.getLogger(str(object)[4:])
    logger.log(logging.INFO, 'Start Process for: '+ path)

    halo = HaloObject.Radio_Halo(object, path, maskpath=maskpath, mask=mask,
                                decreased_fov=decreased_fov,logger=logger, loc=loc,
                                M500=M500, R500=R500, z=z, spectr_index=spectr_index)
    return halo
