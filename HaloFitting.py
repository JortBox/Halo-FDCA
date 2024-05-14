#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
Version: October 2020
'''

from astropy.coordinates import SkyCoord
import logging
import os, sys
from datetime import datetime
import astropy.units as u
import numpy as np
import argparse

import halo_fdca

def arguments():
    parser = argparse.ArgumentParser(description='Halo-FDCA: An automated flux density calculator for radio halos in galaxy clusters. (Boxelaar et al.)')
    parser.add_argument('-object',     help='(str) Cluster object name', type=str, default='Cluster')
    parser.add_argument('-path_in',     help='(str) FITS image location (containing radio halo).', type=str, default='')
    parser.add_argument('-z', '--redshift', help='(float) cluster redshift', type=float, default=0.1)
    parser.add_argument('-model',     help='(str) Model to use. choose from (circle, ellipse, rotated_ellipse, skewed). Default: circle', choices=['circle', 'ellipse', 'rotated_ellipse', 'skewed'], default='circle', type=str)
    parser.add_argument('-frame',     help='(str) Coordinate frame. Default: ICRS', default='icrs', type=str)
    parser.add_argument('-loc',       help="(str) Sky coordinates of cluster. provide coordinates of the form: 'hh mm ss.ss -dd mm ss.s' in hourangle units. Default: None and image centre is chosen.", default = None, type=str)
    parser.add_argument('-m',         help='(bool) choose to include mask or not. If True, -maskPath should be specified. Default: True',default=True, type=str2bool)
    parser.add_argument('-m_file',    help='(str) Mask file location. Default: None', default=None, type=str)
    parser.add_argument('-path_out',  help='(str) Path to code output. Default: directory code is in.', default='./', type=str)
    parser.add_argument('-fov',       help='(bool) Declare if image size has to be decreased before MCMC-ing. Amount of decreasement has ben automatically set to 3.5*r_e. Default: True',default=True, type=str2bool)
    parser.add_argument('-spectr_idx',help='(float) Set spectral index of cluster (S ~ nu^alpha). Used to calculate power and extrapolate flux to arbitrary frequencies. Default: -1.2',default=-1.2, type=float)
    parser.add_argument('-walkers',   help='(int) Number of walkers to deploy in the MCMC algorithm. Default: 200',default=200, type=int)
    parser.add_argument('-steps',     help='(int) Number of evauations each walker has to do. Default: 1200',default=1200, type=int)
    parser.add_argument('-burntime',  help='(int) Burn-in time for MCMC walkers. See emcee documentation for info. Default: None. this is 1/4th of the steps.',default=None, type=int)
    parser.add_argument('-max_radius',help='(float) Maximum posiible radius cut-off. Fitted halos cannot have any r > max_radius. In units of kpc. Default: None (implying image_size/2).',default=None, type=float)
    parser.add_argument('-gamma_prior',help='(bool) Whether to use a gamma distribution as a prior for radii. Default is False. For the gamma parameters: shape = 2.5, scale = 120 kpc. Default: False',default=False, type=str2bool)
    parser.add_argument('-k_exp',     help='(bool) Whether to use k exponent to change shape of exponential distribution. Default: False',default=False, type=str2bool)
    parser.add_argument('-off',       help='(bool) Whether to use an offset in the model (use this when radius is estimated to be too big). Default: False',default=False, type=str2bool)
    parser.add_argument('-s',         help='(bool) Whether to save the mcmc sampler chain in a fits file. Default: True.',default=True, type=str2bool)
    parser.add_argument('-run_mcmc',  help='(bool) Whether to run a MCMC routine or skip it to go straight to processing. can be done if a runned sample already exists in the output path. Default: True',default=True, type=str2bool)
    parser.add_argument('-int_max',   help='(float) Integration radius in r_e units. Default: inf',default=np.inf, type=float)
    parser.add_argument('-freq',      help='(float) frequency in MHz to calculate flux in. When given, the spectral index will be used. Default: image frequency',default=None, type=str)
    parser.add_argument('-rms',      help='(float) Set manual rms noise level to be used by the code in uJy/beam Default: rms calculated by code',default=0., type=float)
    return parser.parse_args()
    
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean type expected.')


def init_logging(args):
    path = args.path_out
    if path[-1]=='/': path = path[:-1]
    now = str(datetime.now())[:19]
    filename = args.path_in.split('/')[-1]
    if not os.path.exists(path+'/log/'):
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
                'filename': path+'/log/'+filename+'_'+now.replace(' ','_')+'.log',
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
    p0     = [halo.I0, halo.centre_pix[0]+diff[0],
              halo.centre_pix[1]+diff[2], r_guess,r_guess,r_guess,r_guess,0.,0.,0.]
    bounds = ([0.,0.,0.,0.,0.,0.,0.,-np.inf, 0., -np.inf],
              [np.inf,halo.data.shape[0],halo.data.shape[1],
               r_bound,r_bound,r_bound,r_bound,np.inf, np.inf, np.inf])
    return p0,bounds


if __name__=='__main__':
    args = arguments()
    
    loc = args.loc
    if loc is not None:
        loc  = SkyCoord(args.loc, unit=(u.hourangle, u.deg), frame=args.frame)

    logging = init_logging(args)
    logger  = logging.getLogger(args.object)
    logger.log(logging.INFO, 'Start Process for: '+ args.object)
    logger.log(logging.INFO, 'Run Arguments: \n'+ str(args)+ '\n')


    Halo = halo_fdca.RadioHalo(args.object, args.path_in, mask_path=args.m_file, mask=args.m,
                            decreased_fov=args.fov,logger=logger, loc=loc,
                            M500=None, R500=None, z=args.redshift,
                            output_path=args.path_out, spectr_index=args.spectr_idx, rms=args.rms)
    
    print(Halo.centre_wcs)
    print(Halo.centre_pix)
    print(Halo.ra)
    sys.exit()
    
    p0, bounds = get_initial_guess(Halo)
    if args.freq is None: 
        args.freq = Halo.freq.value

    if args.run_mcmc:
        fit  = halo_fdca.Fitting(
            Halo, p0, bounds, model=args.model,walkers=args.walkers,steps=args.steps,
            burntime=args.burntime,
            mask=args.m,
            max_radius=args.max_radius,
            gamma_prior=args.gamma_prior,
            k_exponent=args.k_exp, offset=args.off
        )
        #pre_fit_guess = fit.pre_fit()
        fit.run(save=args.s)
    else: pass


    processing = halo_fdca.Processing(
        Halo, args.model,mask=args.m,
        maskpath=args.m_file, save=args.s,
        k_exponent=args.k_exp, offset=False,
        burntime=args.burntime
    )
    
    
    processing.plot_results()
    processing.get_chi2_value()
    frequency = float(args.freq)*u.MHz
    processing.get_flux(int_max=args.int_max, freq=frequency)# error is one sigma (68%).
    processing.get_power(freq=frequency)

    Halo.Close()
