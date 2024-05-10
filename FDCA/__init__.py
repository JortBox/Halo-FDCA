#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
Version: 13 October 2020
'''
import logging
import sys, os
import logging.config
import logging.handlers

from . import HaloObject
from . import markov_chain_monte_carlo
from . import fdca_utils as utils
#from . import plotting_fits

__version__ = '1.0.0'

def Radio_Halo(object, path, decreased_fov=True, maskpath=None, mask=False, loc=None,
                M500=None, R500=None, z=None, outputpath='./', spectr_index=-1.2,
                logger=logging, rms=0):

    halo = HaloObject.Radio_Halo(object, path, maskpath=maskpath, mask=mask,
                                decreased_fov=decreased_fov,logger=logger, loc=loc,
                                M500=M500, R500=R500, z=z,outputpath=outputpath,
                                spectr_index=spectr_index, rms=rms)
    return halo
