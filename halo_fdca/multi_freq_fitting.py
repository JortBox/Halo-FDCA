#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: J.M. Boxelaar
Version: 08 June 2020
"""

from __future__ import division
import sys
import os
import logging
import emcee
import corner

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count, freeze_support, set_start_method
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import gammainc, gamma
from matplotlib.colors import Normalize, LogNorm
from skimage.measure import block_reduce
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.io import fits


# Subfile imports
from . import plot_fits
from . import fdca_utils as utils
from .markov_chain_monte_carlo import *
from .halo_object import RadioHalo

rad2deg = 180.0 / np.pi
deg2rad = np.pi / 180.0
Jydeg2 = u.Jy / (u.deg * u.deg)
mJyarcsec2 = u.mJy / (u.arcsec * u.arcsec)
uJyarcsec2 = 1.0e-3 * u.mJy / (u.arcsec * u.arcsec)


class MultiWavelengthFitting(object):
    """
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
    """

    def __init__(
        self,
        halos: list[RadioHalo],
        p0: list[list] = None,
        bounds: list[tuple[list, list]] = None,
        **kwargs
    ):  
        if p0 is None:
            p0 = [None for _ in halos]
        if bounds is None:
            bounds = [None for _ in halos]
        
        self.halos = halos    
        self.fits: list[Fitting] = list()
        
        for i, halo in enumerate(halos):
            assert isinstance(halo, RadioHalo), "Provide valid RadioHalo object"
            self.fits.append(Fitting(halo, p0=p0[i], bounds=bounds[i], **kwargs))
            print("initiated fitting for %s" % halo.file)
            
        self.walkers = self.fits[0].walkers
        self.steps = self.fits[0].steps
    
            
    def run(self, save=False):
        self.popt = self.pre_fit()
        self.params = np.asarray([fit.params for fit in self.fits])
        self.dim = len(self.params[self.params == True])
        
        halo_info = list()
        for fit in self.fits:
            data = fit.set_data_to_use(fit.data)
            fit.mcmc_noise = utils.findrms(data)
            coord = np.meshgrid(
                np.arange(0,fit.data.shape[1]), np.arange(0,fit.data.shape[0])
            )
            
            # set_dictionary is called to create a dictionary with necessary atributes
            # because 'Pool' cannot pickle the fitting object.
            halo_info.append(set_dictionary(fit))
            
            
        pos = np.asarray([
            self.popt[self.params] * (1.0 + 1.0e-3 * np.random.randn(self.dim))
            for _ in range(self.walkers)
        ])

        sys.exit()
        
        num_CPU = cpu_count()
        with Pool(num_CPU) as pool:
            sampler = emcee.EnsembleSampler(
                self.walkers, 
                self.dim, 
                lnprob, 
                pool=pool, 
                args=[data, coord, halo_info]
            )
            sampler.run_mcmc(pos, self.steps, progress=True)

        self.sampler_chain = sampler.chain
        self.samples = self.sampler_chain[:, int(self.burntime) :, :].reshape(
            (-1, self.dim)
        )

        if save:
            self.save()
            self.plotSampler()

        return self.sampler_chain
            
    def pre_fit(self):
        """
        Do pre fit for all halos, transform the fitted parameters to physical units,
        take the weighted average and use that result as the initial guess for mcmc.
        
        The initial guesses are then transformed back to image units to use in MCMC

        Returns:
            np.ndarray: best fitting params for all images with equal location and radius
        """
        popt = np.zeros((len(self.fits), len(self.fits[0].params)))
        params_unit = np.zeros(popt.shape)
        params_unit_err = np.zeros(popt.shape)
        
        for i, fit in enumerate(self.fits):
            popt[i], perr = fit.pre_fit()

            params_unit[i] = np.asarray(utils.transform_units(fit, np.copy(popt[i])))
            params_unit_err[i] = np.asarray(utils.transform_units(fit, perr, err=True))
            print("pre fit done for %s" % fit.halo.file)

        weight = 1./params_unit_err**2
        params_best = np.sum((params_unit * weight), axis=0)/np.sum(weight, axis=0)
        
        for i, fit in enumerate(self.fits):
            param_sky = wcs.utils.skycoord_to_pixel(
                SkyCoord(params_best[1], params_best[2], unit=u.deg), 
                wcs.WCS(fit.halo.header), origin=1
            )
            popt[i,1] = param_sky[0] - fit.halo.fov_info_mcmc[2]
            popt[i,2] = param_sky[1] - fit.halo.fov_info_mcmc[0]
            
            kpc_scale = 1. / fit.halo.pix2kpc.to(u.kpc).value
            popt[i,3] = params_best[3] * kpc_scale
            
            if fit.modelName in ["ellipse", "rotated_ellipse", "skewed"]:
                popt[i,4] = params_best[4] * kpc_scale

            if fit.modelName == "skewed":
                popt[i,5] = params_best[5] * kpc_scale
                popt[i,6] = params_best[6] * kpc_scale
        
        return popt
            
    def redundant(self, data, _parent_, p0, bounds, model, walkers, steps, burntime, logger, rebin, mask, max_radius, gamma_prior, k_exponent, offset):
        
        if data is None: self.data = [par.data_mcmc for par in _parent_]
        else: self.data: list = data

        if logger is None: self.log = _parent_[0].log
        else: self.log = logger
            
        if p0 is None or bounds is None:
            # change this
            p0_temp, bounds_temp = utils.get_initial_guess(_parent_)
            
        if p0 is None: self.p0 = p0_temp
        else: self.p0 = p0
        
        if bounds is None: self.bounds = bounds_temp
        else: self.bounds = bounds
        
        self.halo = _parent_
        self.orig_shape = [par.data.shape for par in _parent_]
        mask = [par.mask for par in _parent_]
        self.noise = [par.imagenoise for par in _parent_]
        self.rms = [par.rmsnoise for par in _parent_]
        self.sigma = [(self.rms * halo.beam2pix).value for halo in self.halo]
        self.steps = int(steps)
        self.mask_treshold = 0.5
        self.rebin = rebin
        self.k_exponent = k_exponent
        self.offset = offset
        self.gamma_prior = gamma_prior
        

        self.check_settings(model, walkers, mask, burntime, max_radius)
        x = np.arange(0, self.data.shape[1], 1)
        y = np.arange(0, self.data.shape[0], 1)
        self.x_pix, self.y_pix = np.meshgrid(x, y)

        self.dof = len(self.data.value.flat) - self.dim
        return

    def _pre_fit(self):
        # try:
        popt = self.pre_mcmc_fit(
            self.data, p0=np.array(self.p0), bounds=np.array(self.bounds)
        )
        return popt
        # except Exception as e:
        #    self.log.log(logging.CRITICAL,'MCMC Failed to execute pre-fit with error message:\n')
        #    self.log.log(logging.CRITICAL,e)
        #    sys.exit()

    def _run(self, pre_fit_guess=None, save=False):
        data = self.set_data_to_use(self.data)
        x = np.arange(0, self.data.shape[1])
        y = np.arange(0, self.data.shape[0])
        coord = np.meshgrid(x, y)
        
        if pre_fit_guess is None:
            self.popt = self.pre_fit()
        else:
            self.popt = pre_fit_guess
            
        #sys.exit()
        
        self.mcmc_noise = utils.findrms(data)
        pos = [
            self.popt[self.params] * (1.0 + 1.0e-3 * np.random.randn(self.dim))
            for i in range(self.walkers)
        ]
        
        # set_dictionary is called to create a dictionary with necessary atributes
        # because 'Pool' cannot pickle the fitting object.
        halo_info = set_dictionary(self)


        num_CPU = cpu_count()
        with Pool(num_CPU) as pool:
            sampler = emcee.EnsembleSampler(
                self.walkers, self.dim, lnprob, pool=pool, args=[data, coord, halo_info]
            )
            sampler.run_mcmc(pos, self.steps, progress=True)

        self.sampler_chain = sampler.chain
        self.samples = self.sampler_chain[:, int(self.burntime) :, :].reshape(
            (-1, self.dim)
        )

        if save:
            self.save()
            self.plotSampler()

        return self.sampler_chain

    def save(self):
        path = "%s%s_mcmc_samples%s.fits" % (
            self.halo.modelPath,
            self.halo.file.replace(".fits", ""),
            self.filename_append,
        )
        self.hdu = fits.PrimaryHDU()
        self.hdu.data = self.sampler_chain
        self.set_sampler_header()
        self.hdu.writeto(path, overwrite=True)

    def check_settings(self, model, walkers, mask, burntime, max_radius):
        self.modelName = model
        self.paramNames = [
            "I0",
            "x0",
            "y0",
            "r1",
            "r2",
            "r3",
            "r4",
            "ang",
            "k_exp",
            "off",
        ]
        if model == "circle": 
            self._func_ = utils.circle_model
            self._func_mcmc = circle_model
            self.AppliedParameters = [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        elif model == "ellipse":
            self._func_ = utils.ellipse_model
            self._func_mcmc = ellipse_model
            self.AppliedParameters = [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ]
        elif model == "rotated_ellipse":
            self._func_ = utils.rotated_ellipse_model
            self._func_mcmc = rotated_ellipse_model
            self.AppliedParameters = [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                False,
                False,
            ]
        elif model == "skewed":
            self._func_ = utils.skewed_model
            self._func_mcmc = skewed_model
            self.AppliedParameters = [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
            ]
        else:
            self.log.log(logging.CRITICAL, "CRITICAL: invalid model name")
            print("CRITICAL: invalid model name")
            sys.exit()

        if self.k_exponent:
            self.AppliedParameters[-2] = True
        if self.offset:
            self.AppliedParameters[-1] = True

        self.params = pd.DataFrame.from_dict(
            {"params": self.AppliedParameters}, orient="index", columns=self.paramNames
        ).loc["params"]
        self.dim = len(self.params[self.params == True])

        if walkers >= 2 * self.dim:
            self.walkers = int(walkers)
        else:
            self.walkers = int(2 * self.dim + 4)
            self.log.log(
                logging.WARNING,
                "MCMC Too few walkers, nwalkers = {}".format(self.walkers),
            )

        self.image_mask, self.mask = utils.masking(self, mask, full_size=True)
        self.image_mask = self.image_mask[
            self.halo.fov_info_mcmc[0] : self.halo.fov_info_mcmc[1],
            self.halo.fov_info_mcmc[2] : self.halo.fov_info_mcmc[3],
        ]

        if burntime is None:
            self.burntime = int(0.125 * self.steps)
        elif 0.0 > burntime or burntime >= 0.8 * self.steps:
            self.log.log(
                logging.ERROR,
                "MCMC Input burntime of {} is invalid. setting burntime to {}".format(
                    burntime, 0.25 * self.steps
                ),
            )
            self.burntime = int(0.25 * self.steps)
        else:
            self.burntime = int(burntime)

        if max_radius == None:
            self.max_radius = self.data.shape[0] / 2.0
        else:
            self.max_radius = max_radius / self.halo.pix2kpc.value

        filename_append = "_%s" % (self.modelName)
        if self.mask:
            filename_append += "_mask"
        if self.k_exponent:
            filename_append += "_exp"
        if self.offset:
            filename_append += "_offset"
        self.filename_append = filename_append

    def find_mask(self):
        if os.path.isfile(self.halo.maskPath):
            self.mask = True
        else:
            self.mask = False
            self.log.log(logging.ERROR, "No regionfile found,continueing without mask")
    '''
    def set_mask(self):
        regionpath = self.halo.maskPath
        outfile = self.halo.basedir + "Data/Masks/" + self.halo.target + "_mask.fits"
        utils.mask_region(self.halo.path, regionpath, outfile)

        """In 'Radio_Halo', there is a function to decrease the fov of an image. The mask
           is made wrt the entire image. fov_info makes the mask the same shape as
           the image and overlays it"""
        self.image_mask = fits.open(outfile)[0].data[
            0,
            0,
            self.halo.fov_info[0] : self.halo.fov_info[1],
            self.halo.fov_info[2] : self.halo.fov_info[3],
        ]
    '''
    def at(self, parameter):
        par = np.array(self.paramNames)[self.params]
        return np.where(par == parameter)[0][0]

    def set_data_to_use(self, data):
        if self.rebin:
            binned_data = utils.regridding(self.halo, data, decrease_fov=self.halo.cropped)
            if not self.mask:
                self.image_mask = np.zeros(self.data.shape)
                
            self.binned_image_mask = utils.regridding(
                self.halo, 
                self.image_mask * u.Jy, 
                decrease_fov=self.halo.cropped, 
                mask=True
            ).value
            use = binned_data.value
        
            
            return use.ravel()[
                self.binned_image_mask.ravel()
                <= self.mask_treshold * self.binned_image_mask.max()
            ]
        else:
            if self.mask:
                return self.data.value.ravel()[self.image_mask.ravel() <= 0.5]
            else:
                return self.data.value.ravel()

    def pre_mcmc_func(self, obj, *theta):
        theta = utils.add_parameter_labels(obj, theta)
        model = self._func_(obj, theta)
        if obj.mask:
            return model[obj.image_mask.ravel() == 0]
        else:
            return model

    def pre_mcmc_fit(self, image, p0, bounds) -> pd.Series|pd.DataFrame:
        data = image.ravel()

        #p0[1] -= self.halo.margin[2]
        #p0[2] -= self.halo.margin[0]
        if self.mask:
            data = data[self.image_mask.ravel() == 0]

        bounds = (list(bounds[0, self.params]), list(bounds[1, self.params]))
        popt, pcov = curve_fit(
            self.pre_mcmc_func, self, data, p0=tuple(p0[self.params]), bounds=bounds
        )
        perr = np.sqrt(np.diag(pcov))


        #popt[1] += self.halo.margin[2]
        #popt[2] += self.halo.margin[0]
        popt = utils.add_parameter_labels(self, popt)

        if not self.k_exponent:
            popt["k_exp"] = 0.5
        if not self.offset:
            popt["off"] = 0.0

        if self.modelName == "skewed":
            """longest dimension of elliptical shape should always be the x-axis.
            This routine switches x and y if necessary to accomplish this."""
            if (popt["r1"] + popt["r2"]) <= (
                popt["r3"] + popt["r4"]
            ):
                popt["r1"], popt["r3"] = popt["r3"], popt["r1"]
                popt["r2"], popt["r4"] = popt["r4"], popt["r3"]
                popt["ang"] += np.pi / 2.0

        if self.modelName in ["ellipse", "rotated_ellipse"]:
            if popt["r1"] <= popt["r2"]:
                popt["r1"], popt["r2"] = popt["r2"], popt["r1"]
                popt["ang"] += np.pi / 2.0

        if self.modelName in ["rotated_ellipse", "skewed"]:
            """Angle of ellipse from positive x should be between 0 and pi."""
            popt["ang"] = popt["ang"] % (2 * np.pi)
            if popt["ang"] >= np.pi:
                popt["ang"] -= np.pi

        for r in range(4):
            r += 1
            if popt["r" + str(r)] > self.max_radius:
                popt["r" + str(r)] = self.max_radius

        self.centre_pix = np.array([popt["x0"], popt["y0"]], dtype=np.int64)
        self.centre_wcs = wcs.utils.pixel_to_skycoord(self.centre_pix[0], self.centre_pix[1], wcs.WCS(self.halo.header), origin=1)
        
        popt_units = utils.transform_units(self, np.copy(popt))
        popt_units = utils.add_parameter_labels(self, popt_units[self.params])
        self.log.log(
            logging.INFO,
            "MCMC initial guess: \n{} \n and units: muJy/arcsec2, deg, deg, r_e: kpc, rad".format(
                popt_units, perr
            ),
        )

        x = np.arange(0, self.data.shape[1], 1)
        y = np.arange(0, self.data.shape[0], 1)
        self.x_pix, self.y_pix = np.meshgrid(x, y)
        
        return popt

    def plotSampler(self):
        fig, axes = plt.subplots(ncols=1, nrows=self.dim, sharex=True)
        axes[0].set_title("Number of walkers: " + str(self.walkers))
        for axi in axes.flat:
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
            fig.set_size_inches(2 * 10, 15)

        for i in range(self.dim):
            axes[i].plot(
                self.sampler_chain[:, int(self.burntime) :, i].transpose(),
                color="black",
                alpha=0.3,
            )
            axes[i].set_ylabel("param " + str(i + 1), fontsize=15)
            plt.tick_params(labelsize=15)

        plt.savefig(
            "%s%s_walkers%s.pdf"
            % (self.halo.plotPath, self.halo.target, self.filename_append),
            dpi=300,
        )
        plt.clf()
        plt.close(fig)

        labels = list()
        for i in range(self.dim):
            labels.append("Param " + str(i + 1))

        fig = corner.corner(
            self.samples,
            labels=labels,
            quantiles=[0.160, 0.5, 0.840],
            truths=np.asarray(self.popt[self.params]),
            show_titles=True,
            title_fmt=".5f",
        )

        plt.savefig(
            "%s%s_cornerplot%s.pdf"
            % (self.halo.plotPath, self.halo.target, self.filename_append),
            dpi=300,
        )
        plt.clf()
        plt.close(fig)

    def set_sampler_header(self):
        self.hdu.header["nwalkers"] = self.walkers
        self.hdu.header["steps"] = self.steps
        self.hdu.header["dim"] = self.dim
        self.hdu.header["burntime"] = self.burntime
        self.hdu.header["OBJECT"] = (self.halo.name, "Object which was fitted")
        self.hdu.header["IMAGE"] = self.halo.file
        self.hdu.header["UNIT_0"] = ("JY/PIX", "unit of fit parameter")
        self.hdu.header["UNIT_1"] = ("PIX", "unit of fit parameter")
        self.hdu.header["UNIT_2"] = ("PIX", "unit of fit parameter")
        self.hdu.header["UNIT_3"] = ("PIX", "unit of fit parameter")

        if self.dim >= 5:
            self.hdu.header["UNIT_4"] = ("PIX", "unit of fit parameter")
        if self.dim == 8:
            self.hdu.header["UNIT_5"] = ("PIX", "unit of fit parameter")
            self.hdu.header["UNIT_6"] = ("PIX", "unit of fit parameter")
        if self.dim >= 6:
            self.hdu.header["UNIT_7"] = ("RAD", "unit of fit parameter")
        if self.dim == 7:
            self.hdu.header["UNIT_P"] = ("NONE", "unit of fit parameter")

        for i in range(len(self.popt[self.params])):
            self.hdu.header["INIT_" + str(i)] = (
                self.popt[self.params][i],
                "MCMC initial guess",
            )

        self.hdu.header["MASK"] = (self.mask, "was the data masked during fitting")
