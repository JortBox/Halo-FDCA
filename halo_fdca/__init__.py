#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
Version: 13 October 2020
'''
import logging
import logging.config
import logging.handlers

from .halo_object import RadioHalo
#from .markov_chain_monte_carlo import *
from .mcmc_processing import Processing
from .mcmc_fitting_multicomponent import *
from .multi_freq_fitting import MultiWavelengthFitting, MultiWavaelenghtProcessing
from . import fdca_utils as utils
from .fdca_logging import Logger, logger
#from . import plotting_fits

import warnings
warnings.filterwarnings(
    action='ignore',
    module='astropy'
)

__version__ = '1.0.0'
