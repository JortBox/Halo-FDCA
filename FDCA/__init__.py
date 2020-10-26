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

__version__ = '1.0.0'

def Radio_Halo(object, path, decreased_fov=True, maskpath=None, mask=False, loc=None,
                M500=None, R500=None, z=None, outputpath='./', spectr_index=-1.2,
                logger=logging):
                
    halo = HaloObject.Radio_Halo(object, path, maskpath=maskpath, mask=mask,
                                decreased_fov=decreased_fov,logger=logger, loc=loc,
                                M500=M500, R500=R500, z=z,outputpath=outputpath,
                                spectr_index=spectr_index)
    return halo
