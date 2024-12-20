#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: J.M. Boxelaar
Version: 08 June 2020
"""

from __future__ import division
import emcee
import corner

import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits


# Subfile imports
from .. import fdca_utils as utils
from .mcmc_fitting import *
from ..processing import Processing
#from .markov_chain_monte_carlo import *
from ..halo_object import RadioHalo

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
        model: str = "circle",
        **kwargs
    ):  
        if p0 is None:
            p0 = [None for _ in halos]
        if bounds is None:
            bounds = [None for _ in halos]
        
        self.halos = halos    
        self.fits: list[Fitting] = list()
        
        assert model == "circle", "For now, only the circular model is supported in multi frequency mode"
        
        for i, halo in enumerate(halos):
            assert isinstance(halo, RadioHalo), "Provide valid RadioHalo object"
            self.fits.append(Fitting(halo, p0=p0[i], bounds=bounds[i], **kwargs))
            print("initiated fitting for %s" % halo.file)
            
        self.walkers = self.fits[0].walkers
        self.steps = self.fits[0].steps
        self.burntime = self.fits[0].burntime
        
        filename_append = "_%s" % (self.fits[0].model_name)
        if self.fits[0].mask:
            filename_append += "_mask"
        if self.fits[0].k_exponent:
            filename_append += "_exp"
        if self.fits[0].offset:
            filename_append += "_offset"
        self.filename_append = filename_append
    
            
    def run(self, save=False):
        self.popt = self.prepare_fit()
        self.params = np.asarray([fit.params for fit in self.fits])
        self.dim = len(self.params[self.params == True])
        
        halo_info = list()
        data = list()
        coord = list()
        for fit in self.fits:
            #print(fit.data.shape,"data pre flat")
            data_to_use = fit.set_data_to_use(fit.data)
            fit.mcmc_noise = utils.findrms(data_to_use)
            
            #print(data_to_use.shape)
            data.append(data_to_use)
            
            x = np.arange(0, fit.data.shape[1])
            y = np.arange(0, fit.data.shape[0])
            coord.append(np.meshgrid(x, y))
            
            # set_dictionary is called to create a dictionary with necessary atributes
            # because 'Pool' cannot pickle the fitting object.
            halo_info.append(set_dictionary(fit))
        
        parameters_to_use = self.popt[self.params][:5] #hardcoded
        self.dim = len(parameters_to_use)
        
        pos = np.asarray([
            parameters_to_use * (1.0 + 1.0e-3 * np.random.randn(self.dim))
            for _ in range(self.walkers)
        ])
    
        
        num_CPU = cpu_count()
        with Pool(num_CPU) as pool:
            sampler = emcee.EnsembleSampler(
                self.walkers, 
                self.dim, 
                lnprob_multi, 
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
            #self.plotSampler()

        return self.sampler_chain
            
    def prepare_fit(self):
        """
        Do pre fit for all halos, transform the fitted parameters to physical units,
        take the weighted average and use that result as the initial guess for mcmc.
        
        The initial guesses are then transformed back to image units to use in MCMC

        Returns:
            np.ndarray: best fitting params for all images with equal location and radius
        """
        popt = np.zeros((len(self.fits), len(self.fits[0].params)))
        for i, fit in enumerate(self.fits):
            popt[i], perr = fit.pre_fit()
            
        params_unit = np.zeros(popt.shape)
        params_unit_err = np.zeros(popt.shape)
        
        for i, fit in enumerate(self.fits):    
            params_unit_err[i] = np.asarray(utils.transform_units(fit, np.copy(perr), err=True))
        weight = 1./params_unit_err**2
            
        for i, fit in enumerate(self.fits):
            params_unit[i] = np.asarray(utils.transform_units(fit, np.copy(popt[i])))

        params_best = np.sum((params_unit * weight), axis=0)/np.sum(weight, axis=0)
        popt[:,1] = params_best[1]
        popt[:,2] = params_best[2]
        popt[:,3] = params_best[3]
        
        if fit.model_name in ["ellipse", "rotated_ellipse", "skewed"]:
            popt[:,4] = params_best[4]

        if fit.model_name == "skewed":
            popt[:,5] = params_best[5]
            popt[:,6] = params_best[6]   
        return popt
    
    
    def align_params(self, popt: np.ndarray, noise):
        """
        Align the parameters of the halos to the same location and radius.
        """
        params_unit = np.zeros(popt.shape)
        params_unit_err = np.zeros(popt.shape)
        
        if type(noise) == float:
            weight = 1./noise**2
        else:
            for i, fit in enumerate(self.fits):    
                params_unit_err[i] = np.asarray(utils.transform_units(fit, np.copy(noise), err=True))
            weight = 1./params_unit_err**2
            
        for i, fit in enumerate(self.fits):
            params_unit[i] = np.asarray(utils.transform_units(fit, np.copy(popt[i])))

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
            
            if fit.model_name in ["ellipse", "rotated_ellipse", "skewed"]:
                popt[i,4] = params_best[4] * kpc_scale

            if fit.model_name == "skewed":
                popt[i,5] = params_best[5] * kpc_scale
                popt[i,6] = params_best[6] * kpc_scale     
                 
        return popt

    def save(self):
        path = f"{self.halos[0].modelPath}{self.halos[0].name}_multi_mcmc_samples.fits"
        self.hdu = fits.PrimaryHDU()
        self.hdu.data = self.sampler_chain
        self.set_sampler_header()
        self.hdu.writeto(path, overwrite=True)
        
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
            % (self.halos[0].plotPath, self.halos[0].target, self.filename_append),
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
            % (self.halos[0].plotPath, self.halos[0].target, self.filename_append),
            dpi=300,
        )
        plt.clf()
        plt.close(fig)

    def set_sampler_header(self):
        self.hdu.header["nwalkers"] = self.walkers
        self.hdu.header["steps"] = self.steps
        self.hdu.header["dim"] = self.dim
        self.hdu.header["burntime"] = self.burntime
        self.hdu.header["OBJECT"] = (self.halos[0].name, "Object which was fitted")
        self.hdu.header["IMAGE"] = " - ".join([halo.file for halo in self.halos])
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

        self.hdu.header["MASK"] = (" - ".join([str(halo.mask) for halo in self.halos]), "was the data masked during fitting")


class MultiWavaelenghtProcessing(object):
    def __init__(
            self, 
            fits: MultiWavelengthFitting,
            rebin = True,
        ):
        self.halos = fits.halos
        self.n_halos = len(fits.halos)
        self.results: list[Processing] = list()
        
        self.model_name = fits.fits[0].model_name
        self.paramNames = fits.fits[0].paramNames
        self.params = fits.fits[0].params
        
        
        self.extract_chain_file(rebin)
        
        expanded_samples = self.expand_model_params(self.sampler.T)
        pars = np.array(self.paramNames)[self.params]
        n_params = len(pars)
        
        for i, halo in enumerate(self.halos):
            info = self.sample_info.copy()
            for j in range(n_params * self.n_halos):
                del info["INIT_" + str(j)]
            
            for j in range(n_params):
                info["INIT_" + str(j)] = self.sample_info["INIT_" + str(n_params * i + j)]
                
            param_sky = wcs.utils.skycoord_to_pixel(
                SkyCoord(info["INIT_1"], info["INIT_2"], unit=u.deg), 
                halo.wcs, 
                origin=1
            )
            info["INIT_1"] = param_sky[0] - halo.fov_info_mcmc[2]
            info["INIT_2"] = param_sky[1] - halo.fov_info_mcmc[0]
            info["INIT_3"] /= halo.pix2kpc.to(u.kpc).value
            
            self.results.append(Processing(fits.fits[i], sampler=expanded_samples[i].T, sample_info=info))

        
        
    def expand_model_params(self, samples):
        constant_parameters = ["x0", "y0", "r1"] # hardcoded
        pars = np.array(self.paramNames)[self.params]
        n_params = len(pars)
        
        final_theta = np.zeros(tuple([self.n_halos, n_params]) + samples.shape[1:])
        final_theta[0] = samples[:n_params]
        
        compare = np.ones(len(pars), dtype=bool)
        for i in range(len(compare)):
            if pars[i] in constant_parameters:
                compare[i] = False
                
        
        for i in range(self.n_halos-1):
            for par in range(len(compare)):
                if compare[par]:
                    final_theta[i+1, par] = samples[n_params + par]
                else:
                    final_theta[i+1, par] = samples[par]
        
        
        for i, halo in enumerate(self.halos):
            theta = final_theta[i]
            param_sky = wcs.utils.skycoord_to_pixel(
                SkyCoord(theta[1], theta[2], unit=u.deg), 
                halo.wcs, 
                origin=1
            )
            theta[1] = param_sky[0] - halo.fov_info_mcmc[2]
            theta[2] = param_sky[1] - halo.fov_info_mcmc[0]
            
            kpc_scale = 1. / halo.pix2kpc.to(u.kpc).value
            theta[3] = theta[3] * kpc_scale
            
            if self.model_name in ["ellipse", "rotated_ellipse", "skewed"]:
                theta[4] = theta[4] * kpc_scale

            if self.model_name == "skewed":
                theta[5] = theta[5] * kpc_scale
                theta[6] = theta[6] * kpc_scale 
        
        return final_theta
        
    def extract_chain_file(self, rebin):
        self.rebin = rebin
        path = f"{self.halos[0].modelPath}{self.halos[0].name}_multi_mcmc_samples.fits"
        sampler_chain = fits.open(path)

        self.sampler = sampler_chain[0].data
        self.sample_info = sampler_chain[0].header
        
  