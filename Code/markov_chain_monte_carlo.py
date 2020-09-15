#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
Version: 08 June 2020
'''

from __future__ import division
import sys
import os
import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy import ndimage
from scipy.special import gammainc, gamma
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from skimage.measure import block_reduce
from skimage.transform import rescale
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
import emcee
import corner

# Subfile imports
import utils
import plotting_fits as plot

#plt.rc('text',usetex=True)
#plt.rc('font', family='serif')
#np.seterr(divide='ignore', invalid='ignore')

rad2deg    = 180./np.pi
deg2rad    = np.pi/180.
Jydeg2     = u.Jy/(u.deg*u.deg)
mJyarcsec2 = u.mJy/(u.arcsec*u.arcsec)
uJyarcsec2 = 1.e-3*u.mJy/(u.arcsec*u.arcsec)


class fitting(object):
    '''
    -CLASS DESCRIPTION-
    -INPUT-
    _parent_ (Radio_Halo object): Radio_Halo object containing all relevant
                                  object information
    data (2D array): Data array to be fitted. It is adviced to
                         use 'Radio_Halo.data_mcmc'
    dim (int): number of parameters of fitting model to use. Choose from (8,6,5,4).
               Note: currently, only dim=8 works.
    p0 (array like): Initial robust guess for fit parameters. Used for preliminary
                     scipy.optimize.curve_fit. See Scipy documentation for more info.
    bounds (2-tuple of array_like): Initial robust guess for fit parameter bounds.
                                    Used for preliminary scipy.curve_fit. See Scipy
                                    documentation for more info.
    walkers (int): Number of walkers to deploy in the MCMC algorithm
    steps (int): Number of evauations each walker has to do.
    save (bool): Whether to save the mcmc sampler chain in a fits file. default is False
    burntime (int): burn-in time for MCMC walkers. See emcee documentation for info.
    logger: Configured logging object to log info to a .log file. If not given,
            nothing happens.
    rebin (bool): default is True. regridding data to beamsize to fit to indipendent
                  datapoints. Default is True.
    Forward (bool): Depricated.
    Mask (bool): applying mask to image. If true: a DS9 .reg  has to be present in the
                 Radio_halo.maskPath direcory Default is False.
    maskpath (str): Custom path to DS9 region file, read from database.dat.
                    If '--' is given, and mas=True, the standard
                    directory will be searched.
    max_radius (float): maximum posiible radius cut-off. Fitted halos cannot have any
                        r > max_radius. In units of kpc.
                        Default is None (implying image_size/2).
    gamma_prior (bool): wether to use a gamma distribution as a prior for radii.
                        Default is False. For the gamma parameters:
                        shape = 2.5, scale = 120 kpc.
    '''
    def __init__(self, _parent_, data, dim, p0, bounds, walkers, steps,save=False,
                            burntime=0, logger=logging, rebin=True, mask=False,
                            maskpath='--', max_radius=None, gamma_prior=False,
                            k_exponent=False, offset=False):

        if dim not in ['circle','ellipse', 'rotated_ellipse', 'skewed']:
            print('Provide valid function kind')
            sys.exit()

        p0 = list(p0)
        self.orig_shape = _parent_.data.shape
        self.rebin = rebin
        self.log   = logger
        self.halo  = _parent_
        self.noise = _parent_.imagenoise
        self.rms   = _parent_.rmsnoise
        self.data  = data
        self.steps = int(steps)
        self.save  = save
        self.mask_treshold = 0.4
        self.k_exponent = k_exponent
        self.gamma_prior = gamma_prior

        self.p0 = p0
        self.bounds = bounds

        self.check_settings(dim, walkers, mask, burntime, maskpath, max_radius)
        x = np.arange(0,_parent_.data.shape[1],1)
        y = np.arange(0,_parent_.data.shape[0],1)
        self.x_pix, self.y_pix = np.meshgrid(x,y)

        self.dof = len(data.value.flat) - self.dim


    def __preFit__(self):
        try:
            self.pre_mcmc_fit(self.halo.data, p0=np.array(self.p0), bounds=np.array(self.bounds))
        except:
            self.log.log(logging.CRITICAL,'MCMC Failed to execute pre-fit.')
            sys.exit()

    def __run__(self):
        data = self.set_data_to_use(self.data)
        self.mcmc_noise = utils.findrms(data)
        x = np.arange(0, self.data.shape[1])
        y = np.arange(0, self.data.shape[0])
        coord = np.meshgrid(x,y)
        theta_guess = self.popt[self.params]
        pos = [theta_guess*(1.+1.e-3*np.random.randn(self.dim)) for i in range(self.walkers)]

        # set_dictionary is called to create a dictionary with necessary atributes
        # because 'Pool' cannot pickle the fitting object.
        halo_info = set_dictionary(self)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(self.walkers, self.dim, lnprob, pool=pool,
                                            args=[data,coord,halo_info])
            sampler.run_mcmc(pos, self.steps, progress=True)

        self.sampler_chain = sampler.chain
        self.samples = self.sampler_chain[:,int(self.burntime):,:].reshape((-1,self.dim))

        if self.save:
            self.__save__()
            self.plotSampler()

        return self.sampler_chain

    def __save__(self):
        path = '%s%s_mcmc_samples%s.fits' % (self.halo.modelPath,
                                            self.halo.file.replace('.fits',''),
                                            self.filename_append)
        self.hdu      = fits.PrimaryHDU()
        self.hdu.data = self.sampler_chain
        self.set_sampler_header()
        self.hdu.writeto(path, overwrite=True)

    def check_settings(self, dim, walkers, mask, burntime, maskpath, max_radius):
        self.modelName  = dim
        self.paramNames = ['I0','x0','y0','r1','r2','r3','r4','ang','k_exp','off']
        if dim=='circle':
            self._func_ = utils.circle_model
            self._func_mcmc = circle_model
            self.AppliedParameters = [True,True,True,True,False,False,False,False,False,False]
        elif  dim == 'ellipse':
            self._func_ = utils.ellipse_model
            self._func_mcmc = ellipse_model
            self.AppliedParameters = [True,True,True,True,True,False,False,False,False,False]
        elif dim == 'rotated_ellipse':
            self._func_ = utils.rotated_ellipse_model
            self._func_mcmc = rotated_ellipse_model
            self.AppliedParameters = [True,True,True,True,True,False,False,True,False,False]
        elif dim == 'skewed':
            self._func_ = utils.skewed_model
            self._func_mcmc = skewed_model
            self.AppliedParameters = [True,True,True,True,True,True,True,True,False,False]
        else:
            self.log.log(logging.CRITICAL,'CRITICAL: invalid model name')
            print('CRITICAL: invalid model name')
            sys.exit()

        if self.k_exponent: self.AppliedParameters[-2] = True
        if self.offset: self.AppliedParameters[-1]     = True

        self.params = pd.DataFrame.from_dict({'params':self.AppliedParameters},
                                orient='index',columns=self.paramNames).loc['params']
        self.dim    = len(self.params[self.params==True])

        if walkers >= 2*self.dim:
            self.walkers = int(walkers)
        else:
            self.walkers = int(2*self.dim+4)
            self.log.log(logging.WARNING,'MCMC Too few walkers, nwalkers = {}'.format(self.walkers))

        if mask:
            if maskpath == '--':
                self.halo.maskPath = self.halo.basedir+'Data/Masks/'+self.halo.target+'.reg'
            else:
                self.halo.maskPath = maskpath

            self.find_mask()
            if self.mask:
                self.setMask(self.data)
                self.log.log(logging.INFO,'MCMC Mask set')
        else:
            self.log.log(logging.INFO,'MCMC No mask set')
            self.mask=False

        if burntime is None:
            self.burntime = int(0.125*self.steps)
        elif 0. > burntime or burntime >= 0.8*self.steps:
            self.log.log(logging.ERROR,'MCMC Input burntime of {} is invalid. setting burntime to {}'\
                                            .format(burntime, 0.25*self.steps))
            self.burntime = int(0.25*self.steps)
        else:
            self.burntime = int(burntime)

        if max_radius == None:
            self.max_radius = self.data.shape[0]/2.
        else:
            self.max_radius = max_radius/self.halo.pix2kpc

        filename_append = '_%s' % (self.modelName)
        if self.mask: filename_append += '_mask'
        if self.k_exponent: filename_append += '_exp'
        if self.offset: filename_append += '_offset'
        self.filename_append = filename_append

    def find_mask(self):
        if os.path.isfile(self.halo.maskPath):
            self.mask = True
        else:
            self.mask=False
            self.log.log(logging.ERROR,'No regionfile found,continueing without mask')


    def setMask(self, data):
        regionpath = self.halo.maskPath
        outfile    = self.halo.basedir+'Data/Masks/'+self.halo.target+'_mask.fits'
        utils.mask_region(self.halo.path, regionpath, outfile)

        '''In 'Radio_Halo', there is a function to decrease the fov of an image. The mask
           is made wrt the entire image. fov_info makes the mask the same shape as
           the image and overlays it'''
        self.image_mask = fits.open(outfile)[0].data[0,0,
                                self.halo.fov_info[0]:self.halo.fov_info[1],
                                self.halo.fov_info[2]:self.halo.fov_info[3]]

    def set_data_to_use(self,data):
        if self.rebin:
            binned_data = self.halo.regridding(data, decrease_fov=True)
            if not self.mask:
                self.image_mask = np.zeros(self.halo.data.shape)
            self.binned_image_mask = self.halo.regridding(self.image_mask*u.Jy).value
            use = binned_data.value
            #print('rebinned image shape',use.shape)
            #plt.imshow(use)
            #plt.show()
            return use.ravel()[self.binned_image_mask.ravel() <=\
                                    self.mask_treshold*self.binned_image_mask.max()]
        else:
            if self.mask:
                return self.data.value.ravel()[self.image_mask.ravel() <= 0.5]
            else: return self.data.value.ravel()

    def pre_mcmc_func(self, obj, *theta):
        theta = utils.add_parameter_labels(obj, theta)
        model = self._func_(obj, theta)
        if obj.mask:
            return model[obj.image_mask.ravel() == 0]
        else: return model

    def pre_mcmc_fit(self, image, p0, bounds):
        data = image.ravel()
        p0[1]-=self.halo.margin[2]
        p0[2]-=self.halo.margin[0]
        if self.mask:
            data = data[self.image_mask.ravel() == 0]

        bounds = (list(bounds[0,self.params]), list(bounds[1,self.params]))
        popt, pcov = curve_fit(self.pre_mcmc_func,self,data,
                                p0=tuple(p0[self.params]),
                                bounds=bounds)
        perr = np.sqrt(np.diag(pcov))

        #plt.imshow(image.value)
        #plt.contour(self._func_(self,*popt).reshape(image.shape))
        #plt.show()

        popt[1]+= self.halo.margin[2]
        popt[2]+= self.halo.margin[0]
        self.popt = utils.add_parameter_labels(self, popt)
        self.perr = perr

        if not self.k_exponent: self.popt['k_exp'] = 0.5
        if not self.offset:     self.popt['off']   = 0.0

        if self.modelName == 'skewed':
            '''longest dimension of elliptical shape should always be the x-axis.
               This routine switches x and y if necessary to accomplish this.'''
            if (self.popt['r1']+self.popt['r2']) <= (self.popt['r3']+self.popt['r4']):
                self.popt['r1'], self.popt['r3'] = self.popt['r3'], self.popt['r1']
                self.popt['r2'], self.popt['r4'] = self.popt['r4'], self.popt['r3']
                self.popt['ang'] += np.pi/2.

        if self.modelName in ['ellipse','rotated_ellipse']:
            if self.popt['r1']<=self.popt['r2']:
                self.popt['r1'],self.popt['r2'] = self.popt['r2'],self.popt['r1']
                self.popt['ang'] += np.pi/2.

        if self.modelName in ['rotated_ellipse', 'skewed']:
            '''Angle of ellipse from positive x should be between 0 and pi.'''
            self.popt['ang'] = self.popt['ang']%(2*np.pi)
            if self.popt['ang']>=np.pi:
                self.popt['ang'] -= np.pi

        self.centre_pix = np.array([self.popt['x0'],self.popt['y0']], dtype=np.int64)
        self.log.log(logging.INFO,'MCMC initial guess: {} \n with error: {}'\
                                    .format(self.popt[self.params],self.perr))

        x = np.arange(0,self.data.shape[1],1)
        y = np.arange(0,self.data.shape[0],1)
        self.x_pix, self.y_pix = np.meshgrid(x,y)

    def plotSampler(self):
        fig, axes = plt.subplots(ncols=1, nrows=self.dim, sharex=True)
        axes[0].set_title('Number of walkers: '+str(self.walkers))
        for axi in axes.flat:
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
            fig.set_size_inches(2*10,15)

        for i in range(self.dim):
            axes[i].plot(self.sampler_chain[:, int(self.burntime):, i].transpose(),
                                            color='black', alpha=0.3)
            axes[i].set_ylabel('param '+str(i+1), fontsize=15)
            plt.tick_params(labelsize=15)

        plt.savefig('%s%s_walkers%s.png' % (self.halo.plotPath,
                                        self.halo.target,self.filename_append),dpi=300)
        plt.clf()
        plt.close(fig)

        labels = list()
        for i in range(self.dim):
            labels.append('Param '+str(i+1))

        fig = corner.corner(self.samples,labels=labels, quantiles=[0.160, 0.5, 0.840],
                            truths=np.asarray(self.popt[self.params]),
                            show_titles=True, title_fmt='.5f')

        plt.savefig('%s%s_cornerplot%s.png' % (self.halo.plotPath,
                                        self.halo.target,self.filename_append),dpi=300)
        plt.clf()
        plt.close(fig)

    def set_sampler_header(self):
        self.hdu.header['nwalkers'] = (self.walkers)
        self.hdu.header['steps']    = (self.steps)
        self.hdu.header['dim']      = (self.dim)
        self.hdu.header['burntime'] = (self.burntime)
        self.hdu.header['OBJECT']   = (self.halo.name,'Object which was fitted')
        self.hdu.header['IMAGE']    = (self.halo.file)
        self.hdu.header['UNIT_0'] = ('JY/PIX','unit of fit parameter')
        self.hdu.header['UNIT_1'] = ('PIX','unit of fit parameter')
        self.hdu.header['UNIT_2'] = ('PIX','unit of fit parameter')
        self.hdu.header['UNIT_3'] = ('PIX','unit of fit parameter')

        if self.dim>=5:
            self.hdu.header['UNIT_4'] = ('PIX','unit of fit parameter')
        if self.dim == 8:
            self.hdu.header['UNIT_5'] = ('PIX','unit of fit parameter')
            self.hdu.header['UNIT_6'] = ('PIX','unit of fit parameter')
        if self.dim >= 6:
            self.hdu.header['UNIT_7'] = ('RAD','unit of fit parameter')
        if self.dim == 7:
            self.hdu.header['UNIT_P'] = ('NONE','unit of fit parameter')

        for i in range(len(self.popt[self.params])):
            self.hdu.header['INIT_'+str(i)] = (self.popt[self.params][i], 'MCMC initial guess')

        self.hdu.header['MASK'] = (self.mask,'was the data masked during fitting')


def set_dictionary(obj):
    halo_info = {
        "modelName":         obj.modelName,
        "bmaj":              obj.halo.bmaj,
        "bmin":              obj.halo.bmin,
        "bpa":               obj.halo.bpa,
        "pix_size":          obj.halo.pix_size,
        "beam_area":         obj.halo.beam_area,
        "beam2pix":          obj.halo.beam2pix,
        "pix2kpc":           obj.halo.pix2kpc,
        "mask":              obj.mask,
        "sigma":             obj.mcmc_noise,
        "margin":            obj.halo.margin,
        "_func_":            obj._func_mcmc,
        "image_mask":        obj.image_mask,
        "binned_image_mask": obj.binned_image_mask,
        "mask_treshold":     obj.mask_treshold,
        "max_radius":        obj.max_radius,
        "params":            obj.params,
        "paramNames":        obj.paramNames,
        "gamma_prior":       obj.gamma_prior,
        }
    return halo_info

def set_model_to_use(info,data):
    binned_data = regrid_to_beamsize(info, data.value)
    return binned_data.ravel()[info['binned_image_mask'].ravel() <=\
                               info['mask_treshold']*info['binned_image_mask'].max()]

def rotate_image(info,img, decrease_fov=False):
    margin = info['margin']
    img_rot = ndimage.rotate(img, -info['bpa'].value, reshape=False)
    f = img_rot[margin[0]:margin[1], margin[2]:margin[3]]
    #plt.imshow(f)
    #plt.show()
    return f

def regrid_to_beamsize(info, img, accuracy=100.):
    x_scale = np.sqrt(np.pi/(4*np.log(2.)))*obj.bmaj.value
    y_scale = np.sqrt(np.pi/(4*np.log(2.)))*obj.bmin.value

    new_pix_size = np.array((y_scale,x_scale))
    accuracy = int(1./accuracy*100)

    scale = np.round(accuracy*new_pix_size/info['pix_size']).astype(np.int64).value
    pseudo_size = (accuracy*np.array(img.shape) ).astype(np.int64)
    pseudo_array = np.zeros((pseudo_size))

    orig_scale = (np.array(pseudo_array.shape)/np.array(img.shape)).astype(np.int64)
    elements   = np.prod(np.array(orig_scale,dtype='float64'))

    if accuracy is 1:
        pseudo_array = np.copy(img)
    else:
        for j in range(img.shape[0]):
            for i in range(img.shape[1]):
                pseudo_array[orig_scale[1]*i:orig_scale[1]*(i+1),
                             orig_scale[0]*j:orig_scale[0]*(j+1)] = img[i,j]/elements

    f= block_reduce(pseudo_array, block_size=tuple(scale), func=np.sum, cval=0)
    f=np.delete(f, -1, axis=0)
    f=np.delete(f, -1, axis=1)
    #plt.imshow(f)
    #plt.show()
    #print(pseudo_array.shape, scale, f.shape)
    return f

def convolve_with_gaussian(info,data,rotate):
    if rotate:
        data = rotate_image(info,data,decrease_fov=True)

    sigma1 = (info['bmaj']/info['pix_size'])/np.sqrt(8*np.log(2.))
    sigma2 = (info['bmin']/info['pix_size'])/np.sqrt(8*np.log(2.))
    kernel = Gaussian2DKernel(sigma1, sigma2, info['bpa'])
    astropy_conv = convolve(data,kernel,boundary='extend',normalize_kernel=True)
    return astropy_conv

def circle_model(info, coords, theta, rotate=False):
    x,y = coords
    G   = ((x-theta['x0'])**2+(y-theta['y0'])**2)/theta['r1']**2
    Ir  = theta['I0']*np.exp(-G**(0.5+theta['k_exp']))+theta['off']
    return convolve_with_gaussian(info, Ir, rotate)

def ellipse_model(info, coord , theta, rotate=False):
    x,y = coord
    G  = ((x-theta['x0'])/theta['r1'])**2+((y-theta['y0'])/theta['r2'])**2
    Ir = theta['I0']*np.exp(-G**(0.5+theta['k_exp']))+theta['off']
    return convolve_with_gaussian(info, Ir, rotate)

def rotated_ellipse_model(info, coord, theta, rotate=False):
    x,y = coord
    x_rot =  (x-theta['x0'])*np.cos(theta['ang']) + (y-theta['y0'])*np.sin(theta['ang'])
    y_rot = -(x-theta['x0'])*np.sin(theta['ang']) + (y-theta['y0'])*np.cos(theta['ang'])
    G  = (x_rot/theta['r1'])**2.+(y_rot/theta['r2'])**2.
    Ir = theta['I0']*np.exp(-G**(0.5+theta['k_exp']))+theta['off']
    return convolve_with_gaussian(info, Ir, rotate)

def skewed_model(info, coord, theta, rotate=False):
    x,y=coord
    G_pp = G(x, y, theta['I0'],theta['x0'],theta['y0'],theta['r1'],theta['r3'],theta['ang'],  1.,  1.)
    G_mm = G(x, y, theta['I0'],theta['x0'],theta['y0'],theta['r2'],theta['r4'],theta['ang'], -1., -1.)
    G_pm = G(x, y, theta['I0'],theta['x0'],theta['y0'],theta['r1'],theta['r4'],theta['ang'],  1., -1.)
    G_mp = G(x, y, theta['I0'],theta['x0'],theta['y0'],theta['r2'],theta['r3'],theta['ang'], -1.,  1.)
    Ir   = (theta['I0']*(G_pp+G_pm+G_mm+G_mp))
    return convolve_with_gaussian(info, Ir, rotate)

def G(x,y, I0, x0, y0, re_x,re_y, ang, sign_x, sign_y):
    x_rot =  (x-x0)*np.cos(ang)+(y-y0)*np.sin(ang)
    y_rot = -(x-x0)*np.sin(ang)+(y-y0)*np.cos(ang)
    func  = (np.sqrt(sign_x * x_rot)**4.)/(re_x**2.) +\
            (np.sqrt(sign_y * y_rot)**4.)/(re_y**2.)

    exponent = np.exp(-np.sqrt(func))
    exponent[np.where(np.isnan(exponent))]=0.
    return exponent

def lnL(theta, data, coord, info):
    kwargs = {"rotate" : True}
    raw_model = info['_func_'](info,coord,theta,**kwargs)*u.Jy
    model = set_model_to_use(info, raw_model)
    return -np.sum( ((data-model)**2.)/(2*info['sigma']**2.)\
                        + np.log(np.sqrt(2*np.pi)*info['sigma']) )

def lnprior(theta, shape, info):
    prior = -np.inf
    if (theta['I0'] > 0) and (-0.4 < theta['k_exp'] < 19):
        if (0 <= theta['x0'] < shape[1]) and (0 <= theta['y0'] < shape[0]):
            if 0 < theta['r1'] < info['max_radius']:
                if -np.pi/4. < theta['ang'] < 5*np.pi/4.:
                    prior = 0.0
                if not (0 <= theta['r2'] <= theta['r1']):
                    prior = -np.inf

    if prior != -np.inf:
        if info['modelName'] == 'circle':
            radii = np.array([theta['r1']])
        else:
            radii = np.array([theta['r1'],theta['r2']])
        if info['gamma_prior']:
            prior = np.sum(np.log(utils.gamma_dist(radii, 2.5, 120./info['pix2kpc'].value)))
    return prior

def lnprior8(theta, shape, info):
    prior = -np.inf
    if theta['I0']>0 and (0 < theta['x0'] < shape[1]) and (0 < theta['y0'] < shape[0]):
        if theta['r1'] > 0. and theta['r2'] > 0. and theta['r3'] > 0. and theta['r4'] > 0.:
            if (0. < (theta['r3']+theta['r4']) <= (theta['r1']+theta['r2'])) and ((theta['r1']+theta['r2']) < info['max_radius']*2.):
                if -np.pi/4. < theta['ang'] < 5*np.pi/4.:
                    prior =  0.0

    if prior != -np.inf and info['gamma_prior']:
        #guess = 225./info['pix2kpc'] #average based on known sample of halos.
        #prior = -np.sum(1./2*((theta['r1'])**2 + (theta['r2'])**2)/((info['max_radius']/4.)**2))
        radii = np.array([theta['r1'],theta['r2'],theta['r3'],theta['r4']])
        prior = np.sum(np.log(utils.gamma_dist(radii, 2.5, 120./info['pix2kpc'].value)))
    return prior

def lnprob(theta, data, coord, info):
    theta = add_parameter_labels(info['params'], info['paramNames'], theta)
    if info['modelName'] == 'skewed':
        lp = lnprior8(theta, coord[0].shape, info)
    else:
        lp = lnprior(theta, coord[0].shape, info)
    if not np.isfinite(lp):
        return -np.inf
    return lnL(theta, data, coord, info) + lp

def add_parameter_labels(params, paramNames, array):
    full_array = np.zeros(params.shape)
    full_array[params==True] = array
    parameterised_array = pd.DataFrame.from_dict({'params': full_array},
                            orient='index',columns=paramNames).loc['params']
    return parameterised_array







class processing(object):
    '''
    -CLASS DESCRIPTION-
    -INPUT-
    _parent_ (Radio_Halo object): Radio_Halo object containing all relevant
                                  object information
    data (2D array): Data array to be fitted. It is adviced to
                         use 'Radio_Halo.data_mcmc'
    dim (int): number of parameters of fitting model to use. Choose from (8,6,5,4).
               Note: currently, only dim=8 works.
    walkers (int): Number of walkers to deploy in the MCMC algorithm
    steps (int): Number of evauations each walker has to do.
    save (bool): Whether to save the mcmc sampler chain in a fits file. default is False
    burntime (int): burn-in time for MCMC walkers. See emcee documentation for info.
    logger: Configured logging object to log info to a .log file. If not given,
            nothing happens.
    rebin (bool): default is True. regridding data to beamsize to fit to indipendent
                  datapoints. Default is True.
    Forward (bool): Depricated.
    Mask (bool): applying mask to image. If true: a DS9 .reg  has to be present in the
                 Radio_halo.maskPath direcory Default is False.
    maskpath (str): Custom path to DS9 region file, read from database.dat.
                    If '--' is given, and mask=True, the standard
                    directory will be searched.
    '''
    def __init__(self, _parent_, data, dim, logger, save=True, mask=False,
                    rebin=True, maskpath='--', k_exponent=False, offset=False):
        x = np.arange(0,data.shape[1],1)
        y = np.arange(0,data.shape[0],1)
        self.x_pix, self.y_pix = np.meshgrid(x,y)

        self.log = logger
        self.log.log(logging.INFO,'MCMC parameter dimension: {}'.format(dim))
        self.noise = _parent_.imagenoise
        self.rms   = _parent_.rmsnoise
        self.data  = data
        self.save  = save
        self.halo  = _parent_
        self.alpha = _parent_.alpha # spectral index guess
        self.k_exponent = k_exponent
        self.offset = offset
        self.mask_treshold = 0.4

        self.check_settings(dim, mask, maskpath)
        self.extract_chain_file(rebin)
        self.retreive_mcmc_params()
        self.set_labels_and_units()

        self.dof = len(data.value.flat) - self.dim

    def plot_results(self):
        plot.fit_result(self, self.model, self.halo.data,
                              self.halo.rmsnoise, mask=self.mask, regrid=False)
        plot.fit_result(self, self.model, self.halo.data,
                              self.halo.rmsnoise, mask=self.mask,regrid=True)
        #self.plotSampler()
        #self.cornerplot()

    def check_settings(self, dim, mask, maskpath):
        self.modelName  = dim
        self.paramNames = ['I0','x0','y0','r1','r2','r3','r4','ang','k_exp','off']
        if dim=='circle':
            self._func_ = utils.circle_model
            self.AppliedParameters = [True,True,True,True,False,False,False,False,False,False]
        elif  dim == 'ellipse':
            self._func_ = utils.ellipse_model
            self.AppliedParameters = [True,True,True,True,True,False,False,False,False,False]
        elif dim == 'rotated_ellipse':
            self._func_ = utils.rotated_ellipse_model
            self.AppliedParameters = [True,True,True,True,True,False,False,True,False,False]
        elif dim == 'skewed':
            self._func_ = utils.skewed_model
            self.AppliedParameters = [True,True,True,True,True,True,True,True,False,False]
        else:
            self.log.log(logging.CRITICAL,'CRITICAL: invalid model name')
            print('CRITICAL: invalid model name')
            sys.exit()

        if self.k_exponent: self.AppliedParameters[-2] = True
        if self.offset: self.AppliedParameters[-1]     = True

        self.params = pd.DataFrame.from_dict({'params':self.AppliedParameters},
                                orient='index',columns=self.paramNames).loc['params']
        self.dim    = len(self.params[self.params])

        if mask:
            if maskpath == '--':
                self.halo.maskPath = self.halo.basedir+'Data/Masks/'+self.halo.target+'.reg'
            else:
                self.halo.maskPath = maskpath

            fitting.find_mask(self)
            if self.mask:
                fitting.setMask(self,self.data)
                self.log.log(logging.INFO,'MCMC Mask set')
        else:
            self.log.log(logging.INFO,'MCMC No mask set')
            self.mask=False

    def extract_chain_file(self, rebin):
        filename_append = '_{}'.format(self.modelName)
        if self.mask: filename_append += '_mask'
        #if rebin: filename_append += '_rebin'
        if self.k_exponent: filename_append += '_exp'
        if self.offset: filename_append += '_offset'
        self.filename_append = filename_append

        self.rebin = rebin
        sampler_chain = fits.open(self.halo.modelPath+self.halo.file.replace('.fits','')+\
                                        '_mcmc_samples'+self.filename_append+'.fits')

        self.sampler = (sampler_chain[0].data)
        self.info = sampler_chain[0].header

    def at(self, parameter):
        par = np.array(self.paramNames)[self.params]
        return np.where(par==parameter)[0][0]

    def retreive_mcmc_params(self):
        self.walkers  = self.info['nwalkers']
        self.steps    = self.info['steps']
        self.burntime = int(self.info['burntime'])
        self.popt     = utils.add_parameter_labels(self, np.zeros(self.dim))
        try:
            self.noise_mu    = self.halo.header['N_MU']
            self.noise_amp   = self.halo.header['N_AMP']
            self.noise_sigma = self.halo.header['N_SIG']
        except: pass

        for i in range(self.dim):
            self.popt[i] = self.info['INIT_'+str(i)]

        samples = self.sampler[:, self.burntime:, :].reshape((-1, self.dim))

        #translate saples for location to right Fov.
        samples[:,self.at('x0')] -= self.halo.margin[2]
        samples[:,self.at('y0')] -= self.halo.margin[0]

        self.percentiles = self.get_percentiles(samples)
        self.parameters  = utils.add_parameter_labels(self, self.percentiles[:,1].reshape(self.dim))
        self.centre_pix  = np.array([self.parameters['x0'],self.parameters['y0']], dtype=np.int64)

        self.model = self._func_(self, self.parameters)\
                        .reshape(self.x_pix.shape)*u.Jy

        self.samples = samples


    def get_percentiles(self,samples):
        percentiles = np.ones((samples.shape[1],3))
        for i in range(samples.shape[1]):
            percentiles[i,:] = np.percentile(samples[:, i], [16, 50, 84])

        if self.modelName in ['rotated_ellipse', 'skewed']:
            cosine = np.percentile(np.cos(samples[:,self.at('ang')]), [16, 50, 84])
            sine   = np.percentile(np.sin(samples[:,self.at('ang')]), [16, 50, 84])
            arccosine = np.arccos(cosine)
            arcsine   = np.arcsin(sine)

            if arcsine[1] == arccosine[1]:
                ang = arcsine.copy()
            elif arcsine[1] == -arccosine[1]:
                ang = arcsine.copy()
            elif arcsine[1] != arccosine[1] and arcsine[1] != -arccosine[1]:
                if arcsine[1] < 0:
                    ang = -arccosine.copy()
                elif arcsine[1] > 0:
                    ang = arccosine.copy()
            else:
                self.log.log(logging.ERROR,'Angle matching failed in processing.get_percentiles. continueing with default.')
                ang = np.percentile(samples[:,self.at('ang')], [16, 50, 84])

            percentiles[self.at('ang'),:] = ang
        return percentiles

    def cornerplot(self):
        try:
            fig = corner.corner(self.samples_units,labels=self.labels_units,truths=self.popt_units,
                        quantiles=[0.160, 0.5, 0.840], show_titles=True, max_n_ticks=3, title_fmt=self.fmt)
        except:
            fig = corner.corner(self.samples_units,labels=self.labels_units,truths=self.popt_units,
                        quantiles=[0.160, 0.5, 0.840], show_titles=True, max_n_ticks=3, title_fmt='1.2g')
        if self.save:
            plt.savefig(self.halo.plotPath+self.halo.file.replace('.fits','')+'_cornerplot'+self.filename_append+'.pdf')
            plt.clf()
            plt.close(fig)
        else:
            plt.show()

    def plotSampler(self):
        fig, axes = plt.subplots(ncols=1, nrows=self.dim, sharex=True)
        axes[0].set_title('Number of walkers: '+str(self.walkers), fontsize=25)
        for axi in axes.flat:
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
            fig.set_size_inches(2*10,15)

        for i in range(self.dim):
            #axes[i].plot(self.sampler[:, int(0.4*self.sampler.shape[1]):, i].transpose(),color='black', alpha=0.3)
            axes[i].plot(self.sampler[:, :, i].transpose(),color='black', alpha=0.3,lw=0.5)
            axes[i].set_ylabel(self.labels[i], fontsize=20)
            axes[-1].set_xlabel('steps', fontsize=20)
            axes[i].axvline(0.3*self.sampler.shape[1], ls='dashed', color='red')
            axes[i].tick_params(labelsize=20)
            plt.xlim(0, self.sampler.shape[1])

        if self.save:
            plt.savefig(self.halo.plotPath+self.halo.file.replace('.fits','')+'_walkers'+self.filename_append+'.pdf')
            plt.clf()
            plt.close(fig)
        else:
            plt.show()

    def set_labels_and_units(self):
        self.samples_units = self.samples.copy()
        samples_units = self.samples.copy()
        samples_list  = list()

        x0   = np.percentile(self.samples.real[:, 1], [16, 50, 84])[1]-abs(self.halo.margin[1])
        y0   = np.percentile(self.samples.real[:, 2], [16, 50, 84])[1]-abs(self.halo.margin[0])
        self.centre_pix = np.array([x0,y0], dtype=np.int64)
        #print(self.centre_pix, self.halo.dec.shape, self.halo.ra.shape)
        self.centre_wcs = np.array((self.halo.ra.value[self.centre_pix[1]],
                                    self.halo.dec.value[self.centre_pix[0]]))*u.deg

        #print(self.centre_pix,np.percentile(self.samples.real[:, 1], [16, 50, 84])[1])
        #print(self.centre_wcs)

        for i in range(self.dim):
            samples_list.append(samples_units[:,i])

        transformed = self.transform_units(samples_list)
        for i in range(self.dim):
            self.samples_units[:,i] = transformed[i]

        self.popt_units        = self.transform_units(np.copy(self.popt))
        self.percentiles_units = self.get_percentiles(self.samples_units)
        self.params_units      = self.percentiles_units[:,1].reshape(self.dim)
        self.get_units()

    def transform_units(self, params):
        params[0] = ((u.Jy*params[0]/self.halo.pix_area).to(uJyarcsec2)).value
        params[1] = (params[1]-self.centre_pix[0])*self.halo.pix_size.value+self.centre_wcs[0].value
        params[2] = (params[2]-self.centre_pix[1])*self.halo.pix_size.value+self.centre_wcs[1].value
        params[3] = ((params[3]*self.halo.pix2kpc).to(u.kpc)).value
        if self.modelName in ['ellipse', 'rotated_ellipse', 'skewed']:
            params[4] = ((params[4]*self.halo.pix2kpc).to(u.kpc)).value

        if self.modelName == 'skewed':
            params[5] = ((params[5]*self.halo.pix2kpc).to(u.kpc)).value
            params[6] = ((params[6]*self.halo.pix2kpc).to(u.kpc)).value
        if self.modelName in ['rotated_ellipse', 'skewed']:
            params[self.at('ang')] = params[self.at('ang')]
        return params

    def get_units(self):
        labels = ['$I_0$','$x_0$','$y_0$']
        units  = ['$\\mu$Jy arcsec$^{-2}$','deg','deg']
        fmt    = ['.2f','.4f','.4f']

        if self.modelName == 'skewed':
            labels.extend(('$r_{x^+}$','$r_{x^-}$','$r_{y^+}$','$r_{y^-}$'))
            units.extend(('kpc','kpc','kpc','kpc'))
            fmt.extend(('.0f','.0f','.0f','.0f'))
        elif self.modelName in ['ellipse', 'rotated_ellipse']:
            labels.extend(('$r_{x}$','$r_{y}$'))
            units.extend(('kpc','kpc'))
            fmt.extend(('.1f','.1f'))
        elif self.modelName == 'circle':
            labels.append('$r_{e}$')
            units.append('kpc')
            fmt.append('.1f')
        if self.modelName in ['rotated_ellipse', 'skewed']:
            labels.append('$\\phi_e$')
            units.append('Rad')
            fmt.append('.3f')
        if self.k_exponent:
            labels.append('$k$')
            units.append(' ')
            fmt.append('.3f')
        if self.offset:
            labels.append('$C$')
            units.append(' ')
            fmt.append('.3f')

        self.labels = np.array(labels,dtype='<U30')
        self.units  = np.array(units, dtype='<U30')
        self.fmt    = np.array(fmt,   dtype='<U30')

        self.labels_units = np.copy(self.labels)
        for i in range(self.dim):
            self.labels_units[i] = self.labels[i]+' ['+self.units[i]+']'


    def get_confidence_interval(self, percentage=95, units=True):
        alpha   = 1. - percentage/100.
        z_alpha = stats.norm.ppf(1.-alpha/2.)
        se      = np.zeros((self.dim))

        if units:
            for i in range(self.dim):
                se[i] = np.sqrt( np.mean(self.samples_units[:, i]**2.)\
                                -np.mean(self.samples_units[:, i])**2. )
            conf_low = self.params_units-z_alpha*se
            conf_up  = self.params_units+z_alpha*se
            for i in range(self.dim):
                self.log.log(logging.INFO,'{}% Confidence interval of {}: ({:.5f}, {:.5f}) {}'\
                            .format(percentage*100,self.labels[i],conf_low[i],
                                    conf_up[i],self.units[i]))
            self.log.log(logging.INFO,'')
        else:
            for i in range(self.dim):
                se[i] = np.sqrt( np.mean(self.samples[:, i]**2.)\
                                -np.mean(self.samples[:, i])**2. )
            conf_low = self.parameters-z_alpha*se
            conf_up  = self.parameters+z_alpha*se
            for i in range(self.dim):
                self.log.log(logging.INFO,'{}% Confidence interval of {}: ({:.5f}, {:.5f})'\
                            .format(percentage*100,self.labels[i],conf_low[i],
                                    conf_up[i]))
            self.log.log(logging.INFO,'')

        return [conf_low, conf_up]


    def get_chi2_value(self,mask_treshold = 0.4):
        self.mask_treshold = mask_treshold
        x = np.arange(0,self.halo.data_mcmc.shape[1],1)
        y = np.arange(0,self.halo.data_mcmc.shape[0],1)
        self.x_pix, self.y_pix = np.meshgrid(x,y)

        params     = self.parameters.copy()
        params[1] += self.halo.margin[2]
        params[2] += self.halo.margin[0]

        binned_data  = fitting.set_data_to_use(self, self.halo.data_mcmc)
        model        = self._func_(self, params, rotate=True).reshape(self.halo.data.shape)*u.Jy
        binned_model = utils.regrid_to_beamsize(self.halo, model)
        #plt.imshow(binned_model)
        #plt.show()

        if not self.mask:
            self.image_mask = np.zeros(self.halo.data.shape)

        binned_image_mask = self.halo.regridding(self.image_mask*u.Jy).value
        binned_model = binned_model.ravel()[binned_image_mask.ravel() <=\
                                mask_treshold*binned_image_mask.max()]

        chi2 = np.sum( ((binned_data)/(self.rms.value*self.halo.beam2pix))**2. )
        binned_dof    = len(binned_data)-self.dim
        self.chi2_red = chi2/binned_dof

        #print(self.rms.value*self.halo.beam2pix)
        self.rmsregrid = utils.findrms(binned_data)
        #print('NOISE', self.rms*self.halo.beam2pix, self.rmsregrid)
        #alt_noise = utils.findrms((self.halo.data*self.halo.beam2pix).value)
        #print(self.rmsregrid, self.rms.value*self.halo.beam2pix)
        #print(alt_noise)
        chi2 = np.sum( ((binned_data-binned_model)/(self.rmsregrid))**2. )
        binned_dof    = len(binned_data)-self.dim
        self.chi2_red = chi2/binned_dof
        #print(self.chi2_red)
        self.ln_likelihood = np.sum( ((binned_data-binned_model)**2.)/(2*(self.rmsregrid)**2.)\
                            + np.log(np.sqrt(2*np.pi)*self.rmsregrid))
        self.ln_likelihood = -np.sum(((binned_data-binned_model)**2.)/(2*self.rmsregrid**2.)\
                            + np.log(np.sqrt(2*np.pi)*self.rmsregrid) )
        self.AIC  = 2*(self.dim-self.ln_likelihood)
        self.AICc = self.AIC + 2*(self.dim**2.+self.dim)/(len(binned_data)-self.dim-1)
        self.BIC  = self.dim*np.log(len(binned_data))-2*self.ln_likelihood
        ln = np.sum( ((self.halo.data-model)**2.)/(2*(self.rms)**2.)+ np.log(np.sqrt(2*np.pi)*(self.rms.value)))
        self.AICc_whole = 2*(self.dim-ln) + 2*(self.dim**2.+self.dim)/(len(self.halo.data)-self.dim-1)


        self.log.log(logging.INFO,'chi^2_red: {}'.format(self.chi2_red))
        self.log.log(logging.INFO,'AIC: {}'.format(self.AIC))
        self.log.log(logging.INFO,'AICc: {}'.format(self.AICc))
        self.log.log(logging.INFO,'AIC whole: {}'.format(self.AICc_whole))
        self.log.log(logging.INFO,'BIC: {}'.format(self.BIC))
        #print(self.parameters, self.parameters[0]*np.exp(-1.))
        #print(model.max())
        #fig, axes = plt.subplots(ncols=1, nrows=1)
        #axes.imshow(self.halo.data.value)
        #axes.contour(model.value,colors='black', levels=[self.parameters[0]*np.exp(-2.)])
        #plt.show()
        #plt.close(fig)
        #sys.exit()

        x = np.arange(0,self.data.shape[1],1)
        y = np.arange(0,self.data.shape[0],1)
        self.x_pix, self.y_pix = np.meshgrid(x,y)

    def get_flux(self, int_max=np.inf, freq=None):
        if freq is None:
            freq = self.halo.freq

        a = self.samples[:,3]*self.halo.pix_size
        if self.modelName=='skewed':
            b = self.samples[:,5]*self.halo.pix_size
            c = self.samples[:,4]*self.halo.pix_size
            d = self.samples[:,6]*self.halo.pix_size
            factor = (a*b+c*d+a*d+b*c)

        elif self.modelName in ['ellipse','rotated_ellipse']:
            b = self.samples[:,4]*self.halo.pix_size
            factor = 4*a*b
        else:
            factor = 4*a**2
        if self.k_exponent: m = self.samples[:,self.at('k_exp')]+0.5
        else: m=0.5

        I0   = u.Jy*self.samples[:,0]/self.halo.pix_area
        flux = (gamma(1./m)*np.pi*I0/(4*m) * factor * gammainc(1./m, int_max**(2*m))\

                    *(freq/self.halo.freq)**self.alpha).to(u.mJy)

        self.flux      = np.copy(flux)
        self.flux_freq = freq
        self.flux_val  = np.percentile(flux, 50)
        self.flux_err  = ((np.percentile(flux, 84)-np.percentile(flux, 16))/2.)

        #cal = 0.1
        #sub = 0.1 # Osinga et al. 2020

        #self.flux_std = np.sqrt((cal*self.flux_val.value)**2+sub**2+flux_err**2)*u.mJy
        #self.flux_err = np.sqrt((cal*self.flux.value)**2+sub**2+flux_err**2)*u.mJy
        self.log.log(logging.INFO,'MCMC Flux at {:.1f} {}: {:.2f} +/- {:.2f} {}'\
                                    .format(freq.value, freq.unit, self.flux_val.value,
                                    self.flux_err.value,flux.unit))
        self.log.log(logging.INFO,'S/N based on flux {:.2f}'\
                                    .format(self.flux_val.value/self.flux_err.value))

    def get_power(self, freq=None):
        if freq is None:
            freq = self.halo.freq

        d_L   = self.halo.cosmology.luminosity_distance(self.halo.z)
        power = (4*np.pi*d_L**2. *((1.+self.halo.z)**((-1.*self.alpha) - 1.))*\
                                self.flux*((freq/self.flux_freq)**self.alpha)).to(u.W/u.Hz)
        power_std = (4*np.pi*d_L**2. *((1.+self.halo.z)**((-1.*self.alpha) - 1.))*\
                                self.flux_err*((freq/self.flux_freq)**self.alpha)).to(u.W/u.Hz)
        self.power_std = np.percentile(power_std,50)

        cal = 0.1
        sub = 0.1 # Osinga et al. 2020
        self.power     = np.copy(power)
        self.power_val = np.percentile(power,[50])[0]
        power_err = ((np.percentile(power, [84])[0]-np.percentile(power, [16])[0])/2.).value
        self.power_std = np.sqrt((cal*self.power_val.value)**2+sub**2+power_err**2)
        self.log.log(logging.INFO,'Power at {:.1f} {}: ({:.3g} +/- {:.3g}) {}'\
                                        .format(freq.value, freq.unit,
                                        np.percentile(power,[50])[0].value,
                                        (np.percentile(power, [84])[0]-\
                                        np.percentile(power, [16])[0]).value/2.,
                                        power.unit))

    def tableprint(self):
        cal=0.1
        sub=0.1
        file = self.halo.file.replace('.fits','')+'_mcmc_model_ALL.pdf'
        #rms = ((self.rms/self.halo.pix_area).to(uJyarcsec2)).value
        #power = np.copy(self.power.value)/1.e25
        #power16 = np.sqrt((cal*self.power_val.value)**2+sub**2+(self.power_val.value-np.percentile(self.power.value, 16))**2)/1.e25
        #power84 = np.sqrt((cal*self.power_val.value)**2+sub**2+(np.percentile(self.power.value, 84)-self.power_val.value)**2)/1.e25
        flux16 = (self.flux_val.value-np.percentile(self.flux.value, 16))
        flux84 = (np.percentile(self.flux.value, 84)-self.flux_val.value)
        radius = np.argmax(self.parameters[3:7]*self.halo.pix2kpc)+3
        print(radius)

        if self.dim == 4:
            print('%s & $%.2f^{+%.2f}_{-%.2f}$ & $%.2f^{+%.2f}_{-%.2f}$  & $%.f^{+%.f}_{-%.f}$& $%.f$ & $%.3f$ & $%.2f$ & \\ref{fig:%s} \\vspace{0.05cm}\\\\' % (self.halo.name,
            self.flux_val.value, flux84,flux16, self.percentiles_units[0,1],
            self.percentiles_units[0,2]-self.percentiles_units[0,1],
            self.percentiles_units[0,1]-self.percentiles_units[0,0],
            self.percentiles_units[radius,1],
            self.percentiles_units[radius,2]-self.percentiles_units[radius,1],
            self.percentiles_units[radius,1]-self.percentiles_units[radius,0],
            self.AICc, self.chi2_red, self.flux_val/self.flux_err, self.halo.target))
        elif self.dim==6:
            print(' & $%.2f^{+%.2f}_{-%.2f}$ & $%.2f^{+%.2f}_{-%.2f}$  & $%.f^{+%.f}_{-%.f}$& $%.f$ & $%.3f$& $%.2f$& \\vspace{0.05cm}\\\\' % (self.flux_val.value,
            flux84,flux16, self.percentiles_units[0,1],
            self.percentiles_units[0,2]-self.percentiles_units[0,1],
            self.percentiles_units[0,1]-self.percentiles_units[0,0],
            self.percentiles_units[radius,1],
            self.percentiles_units[radius,2]-self.percentiles_units[radius,1],
            self.percentiles_units[radius,1]-self.percentiles_units[radius,0],
            self.AICc, self.chi2_red, self.flux_val/self.flux_err))
        elif self.dim==8:
            print(' & $%.2f^{+%.2f}_{-%.2f}$ & $%.2f^{+%.2f}_{-%.2f}$  & $%.1f^{+%.f}_{-%.f}$& $%.f$ & $%.3f$& $%.2f$& \\vspace{0.1cm}\\\\' % (self.flux_val.value,
            flux84,flux16, self.percentiles_units[0,1],
            self.percentiles_units[0,2]-self.percentiles_units[0,1],
            self.percentiles_units[0,1]-self.percentiles_units[0,0],
            self.percentiles_units[radius,1],
            self.percentiles_units[radius,2]-self.percentiles_units[radius,1],
            self.percentiles_units[radius,1]-self.percentiles_units[radius,0],
            self.AICc, self.chi2_red, self.flux_val/self.flux_err))


#        print('''
#\\begin{figure}[h]
#  \\begin{center}
#	\\vspace{-1.0cm}
#	\\includegraphics[width=1.\\linewidth]{%s}
 # 	\\captionsetup{ margin=0.5cm,labelfont={footnotesize,bf},font=footnotesize}
  #	\\vspace{-0.6cm}
	#	\caption{\\footnotesize \\emph{%s} image with three model overlay's.}
#	\\label{fig:%s}
#\\end{center}
#\\end{figure}''' % (file, self.halo.name, self.halo.target))
