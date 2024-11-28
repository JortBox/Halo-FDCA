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
import time

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count, freeze_support, set_start_method
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import gammainc, gamma
from skimage.measure import block_reduce
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.io import fits


# Subfile imports
from .. import plot_fits
from .. import fdca_utils as utils
from ..halo_object import RadioHalo

try:
    set_start_method("fork")
    freeze_support()
except RuntimeError:
    pass


rad2deg = 180.0 / np.pi
deg2rad = np.pi / 180.0
Jydeg2 = u.Jy / (u.deg * u.deg)
mJyarcsec2 = u.mJy / (u.arcsec * u.arcsec)
uJyarcsec2 = 1.0e-3 * u.mJy / (u.arcsec * u.arcsec)


class Fitting(object):
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
    min_radius (float): minimum posiible radius cut-off. Fitted halos cannot have any
                        r < min_radius. In units of pixels.
                        Default is 0.
    gamma_prior (bool): wether to use a gamma distribution as a prior for radii.
                        Default is False. For the gamma parameters:
                        shape = 2.5, scale = 120 kpc.
    """

    def __init__(
        self,
        _parent_: RadioHalo,
        p0: list = None,
        bounds: tuple[list, list] = None,
        data = None,
        model: str = "circle",
        walkers: int = 100,
        steps: int = 1200,
        burntime=None,
        logger=None,
        rebin=True,
        mask=False,
        max_radius=None,
        min_radius=0,
        gamma_prior=False,
        k_exponent=False,
        offset=False,
    ):
        assert model in ["circle", "ellipse", "rotated_ellipse", "skewed"], "Provide valid function kind"
        
        if data is None: self.data = _parent_.data_mcmc
        else: self.data = data

        if logger is None: self.log = _parent_.log
        else: self.log = logger
            
        if p0 is None or bounds is None:
            p0_temp, bounds_temp = utils.get_initial_guess(_parent_)
            
        if p0 is None: self.p0 = p0_temp
        else: self.p0 = p0
        
        if bounds is None: self.bounds = bounds_temp
        else: self.bounds = bounds
        
        mask = _parent_.mask
        self.orig_shape = _parent_.data.shape
        self.rebin = rebin
        self.halo = _parent_
        self.noise = _parent_.imagenoise
        self.rms = _parent_.rmsnoise
        self.sigma = (self.rms * self.halo.beam2pix).value
        self.steps = int(steps)
        self.mask_treshold = 0.5
        self.k_exponent = k_exponent
        self.offset = offset
        self.gamma_prior = gamma_prior

        self.check_settings(model, walkers, mask, burntime, max_radius, min_radius)
        x = np.arange(0, self.data.shape[1], 1)
        y = np.arange(0, self.data.shape[0], 1)
        self.x_pix, self.y_pix = np.meshgrid(x, y)

        self.dof = len(self.data.value.flat) - self.dim
        return

    def pre_fit(self):
        popt, perr = self.pre_mcmc_fit(
            self.data, p0=np.array(self.p0), bounds=np.array(self.bounds)
        )
        return popt, perr


    def run(self, pre_fit_guess=None, save=False):
        data = self.set_data_to_use(self.data)
        x = np.arange(0, self.data.shape[1])
        y = np.arange(0, self.data.shape[0])
        coord = np.meshgrid(x, y)
        
        if pre_fit_guess is None:
            self.popt, __ = self.pre_fit()
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
        
        #print(self.data.shape, "data pre flat")
        #print(self.binned_image_mask.shape, "binned_image_mask pre")

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

    def check_settings(self, model, walkers, mask, burntime, max_radius, min_radius):
        self.model_name = model
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
        self.image_mask, self.mask = utils.masking(self, mask, full_size=True)

        if walkers >= 2 * self.dim:
            self.walkers = int(walkers)
        else:
            self.walkers = int(2 * self.dim + 4)
            self.log.log(
                logging.WARNING,
                "MCMC Too few walkers, nwalkers = {}".format(self.walkers),
            )

        self.image_mask, self.mask = utils.masking(self, mask, full_size=True)
        
        if np.any(np.asarray(self.halo.fov_info_mcmc) < 0):
            self.image_mask, fov_info = utils.pad_image(self.image_mask)
        else:
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

        if max_radius == None or max_radius < 0 :
            self.max_radius = self.data.shape[0] / 2.0
        else:
            self.max_radius = max_radius / self.halo.pix2kpc.value
            
        if min_radius <= 0.:
            self.min_radius = 0
        else:
            self.min_radius = min_radius

        filename_append = "_%s" % (self.model_name)
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
            #print(binned_data.shape, "binned_data (data)")
        
            
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

    def pre_mcmc_fit(self, image, p0, bounds):
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
        perr = utils.add_parameter_labels(self, perr)

        if not self.k_exponent:
            popt["k_exp"] = 0.5
        if not self.offset:
            popt["off"] = 0.0

        if self.model_name == "skewed":
            """longest dimension of elliptical shape should always be the x-axis.
            This routine switches x and y if necessary to accomplish this."""
            if (popt["r1"] + popt["r2"]) <= (
                popt["r3"] + popt["r4"]
            ):
                popt["r1"], popt["r3"] = popt["r3"], popt["r1"]
                popt["r2"], popt["r4"] = popt["r4"], popt["r3"]
                popt["ang"] += np.pi / 2.0

        if self.model_name in ["ellipse", "rotated_ellipse"]:
            if popt["r1"] <= popt["r2"]:
                popt["r1"], popt["r2"] = popt["r2"], popt["r1"]
                popt["ang"] += np.pi / 2.0

        if self.model_name in ["rotated_ellipse", "skewed"]:
            """Angle of ellipse from positive x should be between 0 and pi."""
            popt["ang"] = popt["ang"] % (2 * np.pi)
            if popt["ang"] >= np.pi:
                popt["ang"] -= np.pi

        for r in range(4):
            r += 1
            if popt["r" + str(r)] > self.max_radius:
                popt["r" + str(r)] = self.max_radius
            
            if popt["r" + str(r)] < self.min_radius:
                popt["r" + str(r)] = self.min_radius

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
        
        return popt, perr

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


def set_dictionary(obj):
    halo_info = {
        "model_name": obj.model_name,
        "bmaj": obj.halo.bmaj,
        "bmin": obj.halo.bmin,
        "bpa": obj.halo.bpa,
        "pix_size": obj.halo.pix_size,
        "beam_area": obj.halo.beam_area,
        "beam2pix": obj.halo.beam2pix,
        "pix2kpc": obj.halo.pix2kpc,
        "mask": obj.mask,
        "sigma": obj.mcmc_noise,
        "margin": obj.halo.margin,
        "_func_": obj._func_mcmc,
        "image_mask": obj.image_mask,
        "binned_image_mask": obj.binned_image_mask,
        "mask_treshold": obj.mask_treshold,
        "max_radius": obj.max_radius,
        "min_radius": obj.min_radius,
        "params": obj.params,
        "paramNames": obj.paramNames,
        "gamma_prior": obj.gamma_prior,
        "fov_info_mcmc": obj.halo.fov_info_mcmc,
        "wcs": obj.halo.wcs,
        "cropped": obj.halo.cropped,
    }
    return halo_info


def set_model_to_use(info, data):
    #print(data.shape,"data")
    binned_data = regrid_to_beamsize(info, data.value)
    #print(binned_data.shape, "binned_data (model)")
    #print(info["binned_image_mask"].shape, "binned_image_mask")
    return binned_data.ravel()[
        info["binned_image_mask"].ravel()
        <= info["mask_treshold"] * info["binned_image_mask"].max()
    ]


def rotate_image(info, img, decrease_fov=False):
    if info["mask"]: cval=1
    else: cval=0

    if decrease_fov:
        img_rot = ndimage.rotate(
            img, -info["bpa"].value, reshape=False,mode='constant',cval=cval
        )
        f = img_rot[
            info["margin"][2]:info["margin"][3], 
            info["margin"][0]:info["margin"][1]
        ]
        return f
    else:
        if np.array(img.shape)[0]%2 == 0:
            img = np.delete(img, 0, 0)
        if np.array(img.shape)[1]%2 == 0:
            img = np.delete(img, 0, 1)

        pivot = (np.array(img.shape)/2).astype(np.int64)
        padX = [int(img.shape[1]) - pivot[0], pivot[0]]
        padY = [int(img.shape[0]) - pivot[1], pivot[1]]
        img_pad = np.pad(img, [padY, padX], 'constant', constant_values=(cval))
        img_rot = ndimage.rotate(
            img_pad, -info["bpa"].value, reshape=False,mode='constant',cval=cval
        )
        return img_rot[padY[0]:-padY[1], padX[0]:-padX[1]]
    
        

def regrid_to_beamsize(info, img, accuracy=100.0):
    x_scale = np.sqrt(np.pi / (4 * np.log(2.0))) * info["bmaj"].value
    y_scale = np.sqrt(np.pi / (4 * np.log(2.0))) * info["bmin"].value

    new_pix_size = np.array((y_scale, x_scale))
    accuracy = int(1.0 / accuracy * 100)

    scale = np.round(accuracy * new_pix_size / info["pix_size"]).astype(np.int64).value
    pseudo_size = (accuracy * np.array(img.shape)).astype(np.int64)
    pseudo_array = np.zeros((pseudo_size))

    orig_scale = (np.array(pseudo_array.shape) / np.array(img.shape)).astype(np.int64)
    elements = np.prod(np.array(orig_scale, dtype="float64"))

    if accuracy == 1:
        pseudo_array = np.copy(img)
    else:
        for j in range(img.shape[0]):
            for i in range(img.shape[1]):
                pseudo_array[
                    orig_scale[1] * i : orig_scale[1] * (i + 1),
                    orig_scale[0] * j : orig_scale[0] * (j + 1),
                ] = (
                    img[i, j] / elements
                )
                
    f = block_reduce(pseudo_array, block_size=tuple(scale), func=np.sum, cval=0) #type:ignore
    f = np.delete(f, -1, axis=0)
    f = np.delete(f, -1, axis=1)
    return f


def convolve_with_gaussian(info, data, rotate):
    if rotate:
        data = rotate_image(info, data, decrease_fov=info["cropped"])

    sigma1 = (info["bmaj"] / info["pix_size"]) / np.sqrt(8 * np.log(2.0))
    sigma2 = (info["bmin"] / info["pix_size"]) / np.sqrt(8 * np.log(2.0))
    kernel = Gaussian2DKernel(sigma1, sigma2, info["bpa"])
    astropy_conv = convolve(data, kernel, boundary="extend", normalize_kernel=True)
    return astropy_conv


def circle_model(info, coords, theta, rotate=False):
    x, y = coords
    G = ((x - theta["x0"]) ** 2 + (y - theta["y0"]) ** 2) / theta["r1"] ** 2
    Ir = theta["I0"] * np.exp(-(G ** (0.5 + theta["k_exp"]))) + theta["off"]
    #print(Ir.shape, "Ir")
    return convolve_with_gaussian(info, Ir, rotate)


def ellipse_model(info, coord, theta, rotate=False):
    x, y = coord
    G = ((x - theta["x0"]) / theta["r1"]) ** 2 + ((y - theta["y0"]) / theta["r2"]) ** 2
    Ir = theta["I0"] * np.exp(-(G ** (0.5 + theta["k_exp"]))) + theta["off"]
    return convolve_with_gaussian(info, Ir, rotate)


def rotated_ellipse_model(info, coord, theta, rotate=False):
    x, y = coord
    x_rot = (x - theta["x0"]) * np.cos(theta["ang"]) + (y - theta["y0"]) * np.sin(
        theta["ang"]
    )
    y_rot = -(x - theta["x0"]) * np.sin(theta["ang"]) + (y - theta["y0"]) * np.cos(
        theta["ang"]
    )
    G = (x_rot / theta["r1"]) ** 2.0 + (y_rot / theta["r2"]) ** 2.0
    Ir = theta["I0"] * np.exp(-(G ** (0.5 + theta["k_exp"]))) + theta["off"]
    return convolve_with_gaussian(info, Ir, rotate)


def skewed_model(info, coord, theta, rotate=False):
    x, y = coord
    G_pp = G(
        x,
        y,
        theta["I0"],
        theta["x0"],
        theta["y0"],
        theta["r1"],
        theta["r3"],
        theta["ang"],
        1.0,
        1.0,
    )
    G_mm = G(
        x,
        y,
        theta["I0"],
        theta["x0"],
        theta["y0"],
        theta["r2"],
        theta["r4"],
        theta["ang"],
        -1.0,
        -1.0,
    )
    G_pm = G(
        x,
        y,
        theta["I0"],
        theta["x0"],
        theta["y0"],
        theta["r1"],
        theta["r4"],
        theta["ang"],
        1.0,
        -1.0,
    )
    G_mp = G(
        x,
        y,
        theta["I0"],
        theta["x0"],
        theta["y0"],
        theta["r2"],
        theta["r3"],
        theta["ang"],
        -1.0,
        1.0,
    )
    Ir = theta["I0"] * (G_pp + G_pm + G_mm + G_mp)
    return convolve_with_gaussian(info, Ir, rotate)


def G(x, y, I0, x0, y0, re_x, re_y, ang, sign_x, sign_y):
    x_rot = (x - x0) * np.cos(ang) + (y - y0) * np.sin(ang)
    y_rot = -(x - x0) * np.sin(ang) + (y - y0) * np.cos(ang)
    func = (np.sqrt(sign_x * x_rot) ** 4.0) / (re_x**2.0) + (
        np.sqrt(sign_y * y_rot) ** 4.0
    ) / (re_y**2.0)

    exponent = np.exp(-np.sqrt(func))
    exponent[np.where(np.isnan(exponent))] = 0.0
    return exponent


def lnL(theta, data, coord, info):
    kwargs = {"rotate": True}
    raw_model = info["_func_"](info, coord, theta, **kwargs) * u.Jy
    
    model = set_model_to_use(info, raw_model)
    return -0.5 * np.sum(0.5 * ((data - model) / info["sigma"])**2.0) - len(data) * np.log(np.sqrt(2 * np.pi) * info["sigma"])

def lnprior(theta, shape, info):
    prior = -np.inf
    if (theta["I0"] > 0) and (-0.4 < theta["k_exp"] < 19):
        if (0 <= theta["x0"] < shape[1]) and (0 <= theta["y0"] < shape[0]):
            if info["min_radius"] < theta["r1"] < info["max_radius"]:
                if -np.pi * 0.25 < theta["ang"] < 5 * np.pi * 0.25:
                    prior = 0.0
                if not (0 <= theta["r2"] <= theta["r1"]):
                    prior = -np.inf

    if prior != -np.inf:
        if info["model_name"] == "circle":
            radii = np.array([theta["r1"]])
        else:
            radii = np.array([theta["r1"], theta["r2"]])
        if info["gamma_prior"]:
            prior = np.sum(
                np.log(utils.gamma_dist(radii, 2.3, 120.0 / info["pix2kpc"].value))
            )
    return prior


def lnprior8(theta, shape, info):
    prior = -np.inf
    if (
        theta["I0"] > 0
        and (0 < theta["x0"] < shape[1])
        and (0 < theta["y0"] < shape[0])
    ):
        if (
            theta["r1"] > 0.0
            and theta["r2"] > 0.0
            and theta["r3"] > 0.0
            and theta["r4"] > 0.0
        ):
            if (0.0 < (theta["r3"] + theta["r4"]) <= (theta["r1"] + theta["r2"])) and (
                (theta["r1"] + theta["r2"]) < info["max_radius"] * 2.0
            ):
                if -np.pi * 0.25 < theta["ang"] < 5 * np.pi * 0.25:
                    prior = 0.0

    if prior != -np.inf and info["gamma_prior"]:
        # guess = 225./info['pix2kpc'] #average based on known sample of halos.
        # prior = -np.sum(1./2*((theta['r1'])**2 + (theta['r2'])**2)/((info['max_radius']/4.)**2))
        radii = np.array([theta["r1"], theta["r2"], theta["r3"], theta["r4"]])
        prior = np.sum(
            np.log(utils.gamma_dist(radii, 2.3, 120.0 / info["pix2kpc"].value))
        )
    return prior


def lnprob(theta, data, coord, info):
    #time1 = time.time()
    theta = add_parameter_labels(info["params"], info["paramNames"], theta)
    if info["model_name"] == "skewed":
        lp = lnprior8(theta, coord[0].shape, info)
    else:
        lp = lnprior(theta, coord[0].shape, info)
    if not np.isfinite(lp):
        return -np.inf
    
    
    likelihood = lnL(theta, data, coord, info) + lp
    #time2 = time.time()
    #print(f"Time to expand model parameters:", time2 - time1, coord[0].shape)
    return likelihood


def lnprob_multi(theta, data, coord, info):
    #time1 = time.time()
    num_halos = int(len(data))
    theta_split = expand_model_params(theta, info)
    
    likelihood = 0.0
    
    for i in range(num_halos):
        theta = add_parameter_labels(
            info[i]["params"], info[i]["paramNames"], theta_split[i]
        )
        
        if info[i]["model_name"] == "skewed":
            lp = lnprior8(theta, coord[i][0].shape, info[i])
        else:
            lp = lnprior(theta, coord[i][0].shape, info[i])
        if not np.isfinite(lp):
            return -np.inf
        
        
        likelihood += lnL(theta, data[i], coord[i], info[i])
    #time2 = time.time()
    #print(f"Time to expand model parameters:", time2 - time1)
    
    return likelihood + lp

def expand_model_params(theta, info):
    constant_parameters = ["x0", "y0", "r1"] # hardcoded
    pars = np.array(info[0]["paramNames"])[info[0]["params"]]
    n_params = len(pars)
    
    final_theta = np.zeros((len(info), n_params))
    final_theta[0] = theta[:n_params]
    
    
    compare = np.ones(len(pars), dtype=bool)
    for i in range(len(compare)):
        if pars[i] in constant_parameters:
            compare[i] = False
            
    
    for i in range(len(info)-1):
        for par in range(len(compare)):
            if compare[par]:
                final_theta[i+1, par] = theta[n_params + par]
            else:
                final_theta[i+1, par] = theta[par]
    
    for i, fit in enumerate(info):
        theta = final_theta[i]
        param_sky = wcs.utils.skycoord_to_pixel(
            SkyCoord(theta[1], theta[2], unit=u.deg), 
            fit["wcs"], 
            origin=1
        )
        theta[1] = param_sky[0] - fit["fov_info_mcmc"][2]
        theta[2] = param_sky[1] - fit["fov_info_mcmc"][0]
        
        kpc_scale = 1. / fit["pix2kpc"].to(u.kpc).value
        theta[3] = theta[3] * kpc_scale
        
        if fit["model_name"] in ["ellipse", "rotated_ellipse", "skewed"]:
            theta[4] = theta[4] * kpc_scale

        if fit["model_name"] == "skewed":
            theta[5] = theta[5] * kpc_scale
            theta[6] = theta[6] * kpc_scale 
    
    return final_theta

def add_parameter_labels(params, paramNames, array):
    full_array = np.zeros(params.shape)
    full_array[params == True] = array
    parameterised_array = pd.DataFrame.from_dict(
        {"params": full_array}, orient="index", columns=paramNames
    ).loc["params"]
    return parameterised_array


class Processing(object):
    """
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
    """

    def __init__(
        self,
        _parent_: RadioHalo,
        data = None,
        model="circle",
        logger=None,
        save=True,
        mask=False,
        rebin=True,
        maskpath="--",
        k_exponent=False,
        offset=False,
        burntime=None,
        sampler=None,
        info=None,
    ):
        assert model in ["circle", "ellipse", "rotated_ellipse", "skewed"], "Provide valid function kind"

        if data is None:
            self.data = _parent_.data
        else:
            self.data = data

        if logger is None:
            self.log = _parent_.log
        else:
            self.log = logger
        
        x = np.arange(0, self.data.shape[1], 1)
        y = np.arange(0, self.data.shape[0], 1)
        self.x_pix, self.y_pix = np.meshgrid(x, y)

        self.log.log(logging.INFO, "Model name: {}".format(model))
        self.noise = _parent_.imagenoise
        self.rms = _parent_.rmsnoise
        self.save = save
        self.halo = _parent_
        self.alpha = _parent_.alpha  # spectral index guess
        self.k_exponent = k_exponent
        self.offset = offset
        self.mask_treshold = 0.5
        self.int_max = 0
        mask = _parent_.mask

        self.check_settings(model, mask)
        self.extract_chain_file(rebin, sampler, info)
        self.retreive_mcmc_params()
        self.set_labels_and_units()

        self.dof = len(self.data.value.flat) - self.dim
        
    def __repr__(self) -> str:
        
        uncertainties1 = self.percentiles_units[:, 1] - self.percentiles_units[:, 0]
        uncertainties2 = self.percentiles_units[:, 2] - self.percentiles_units[:, 1]

        flux, flux_err = self.get_flux()
        
        param_string = ""
        for param in range(len(self.params[self.params])):
            param_string += f"{self.paramNames[param]}:   {self.params_units[self.params][param]:.5f} ({self.units[param]})\n    "
        
        run_details = f"""
Run information for object {self.halo.name}:
    RMS noise: {self.rms}
    Model: {self.model_name}
    Walkers: {self.walkers}
    Steps: {self.steps}
    Burntime: {self.burntime}
    Mask: {self.mask}
    Rebin: {self.rebin}
    K_exponent: {self.k_exponent}
    Offset: {self.offset}

Fit results:
    Flux density at {self.halo.freq:.1f}: {flux:.5f} +/- {flux_err:.5f} (integrated up to {self.int_max} r_e)
    Reduced chi-squared: {self.get_chi2_value()}
    {param_string}
    Uncertainties (lower, upper):
        {uncertainties1}
        {uncertainties2}
    """
        return run_details

    def plot(self):
        plot_fits.fit_result(
            self,
            self.model,
            self.halo.data,
            self.halo.rmsnoise,
            mask=self.mask,
            regrid=False,
        )
        plot_fits.fit_result(
            self,
            self.model,
            self.halo.data,
            self.halo.rmsnoise,
            mask=self.mask,
            regrid=True,
        )
        self.plotSampler()
        self.cornerplot()

    def check_settings(self, model, mask):
        self.model_name = model
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
        self.dim = len(self.params[self.params])
        self.image_mask = np.zeros(self.halo.data.shape)

        self.image_mask, self.mask = utils.masking(self, mask)
        """
        if mask:
            if maskpath == '--':
                self.halo.maskPath = self.halo.basedir+'Output/'+self.halo.target+'.reg'
            else:
                self.halo.maskPath = maskpath

            fitting.find_mask(self)
            if self.mask:
                fitting.setMask(self,self.data)
                self.log.log(logging.INFO,'MCMC Mask set')
        else:
            self.log.log(logging.INFO,'MCMC No mask set')
            self.mask=False
        """

    def extract_chain_file(self, rebin, sampler, info):
        filename_append = "_{}".format(self.model_name)
        if self.mask:
            filename_append += "_mask"
        # if rebin: filename_append += '_rebin'
        if self.k_exponent:
            filename_append += "_exp"
        if self.offset:
            filename_append += "_offset"
        self.filename_append = filename_append

        self.rebin = rebin
        
        if sampler is not None and info is not None:
            self.sampler = sampler
            self.info = info
            
        else:
            sampler_chain = fits.open(
                self.halo.modelPath
                + self.halo.file.replace(".fits", "")
                + "_mcmc_samples"
                + self.filename_append
                + ".fits"
            )

            self.sampler = sampler_chain[0].data
            self.info = sampler_chain[0].header
            


    def at(self, parameter):
        par = np.array(self.paramNames)[self.params]
        return np.where(par == parameter)[0][0]

    def retreive_mcmc_params(self):
        self.walkers = self.info["nwalkers"]
        self.steps = self.info["steps"]
        burntime = int(self.info["burntime"])
        self.popt = utils.add_parameter_labels(self, np.zeros(self.dim))

        for i in range(self.dim):
            self.popt[i] = self.info["INIT_" + str(i)]

        if burntime is None:
            self.burntime = int(0.25 * self.steps)
        elif 0.0 > burntime or burntime >= self.steps:
            self.log.log(
                logging.ERROR,
                "MCMC Input burntime of {} is invalid. setting burntime to {}".format(
                    burntime, 0.25 * self.steps
                ),
            )
            self.burntime = int(0.25 * self.steps)
        else:
            self.burntime = int(burntime)

        samples = self.sampler[:, self.burntime :, :].reshape((-1, self.dim))

        # translate saples for location to right Fov.
        samples[:, self.at("x0")] -= self.halo.margin[2]
        samples[:, self.at("y0")] -= self.halo.margin[0]

        self.percentiles = self.get_percentiles(samples)
        self.parameters = utils.add_parameter_labels(
            self, self.percentiles[:, 1].reshape(self.dim)
        )
        self.centre_pix = np.array(
            [self.parameters["x0"], self.parameters["y0"]], dtype=np.int64
        )

        self.model = self._func_(self, self.parameters).reshape(self.x_pix.shape) * u.Jy

        self.samples = samples

    def get_percentiles(self, samples):
        percentiles = np.ones((samples.shape[1], 3))
        for i in range(samples.shape[1]):
            percentiles[i, :] = np.percentile(samples[:, i], [16, 50, 84])

        if self.model_name in ["rotated_ellipse", "skewed"]:
            cosine = np.percentile(np.cos(samples[:, self.at("ang")]), [16, 50, 84])
            sine = np.percentile(np.sin(samples[:, self.at("ang")]), [16, 50, 84])
            arccosine = np.arccos(cosine)
            arcsine = np.arcsin(sine)

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
                self.log.log(
                    logging.ERROR,
                    "Angle matching failed in processing.get_percentiles. continueing with default.",
                )
                ang = np.percentile(samples[:, self.at("ang")], [16, 50, 84])

            percentiles[self.at("ang"), :] = ang
        return percentiles

    def cornerplot(self):
        try:
            fig = corner.corner(
                self.samples_units,
                labels=self.labels_units,
                truths=self.popt_units[self.params],
                quantiles=[0.160, 0.5, 0.840],
                show_titles=True,
                max_n_ticks=3,
                title_fmt=self.fmt,
            )
        except:
            fig = corner.corner(
                self.samples_units,
                labels=self.labels_units,
                truths=self.popt_units[self.params],
                quantiles=[0.160, 0.5, 0.840],
                show_titles=True,
                max_n_ticks=3,
                title_fmt="1.2g",
            )
        if self.save:
            plt.savefig(
                self.halo.plotPath
                + self.halo.file.replace(".fits", "")
                + "_cornerplot"
                + self.filename_append
                + ".pdf"
            )
            plt.clf()
            plt.close(fig)
        else:
            plt.show()

    def plotSampler(self):
        fig, axes = plt.subplots(ncols=1, nrows=self.dim, sharex=True)
        axes[0].set_title("Number of walkers: " + str(self.walkers), fontsize=25)
        for axi in axes.flat:
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
            fig.set_size_inches(2 * 10, 15)

        for i in range(self.dim):
            axes[i].plot(
                self.sampler[:, :, i].transpose(), color="black", alpha=0.3, lw=0.5
            )
            axes[i].set_ylabel(self.labels[i], fontsize=20)
            axes[-1].set_xlabel("steps", fontsize=20)
            axes[i].axvline(0.3 * self.sampler.shape[1], ls="dashed", color="red")
            axes[i].tick_params(labelsize=20)
            plt.xlim(0, self.sampler.shape[1])

        if self.save:
            plt.savefig(
                self.halo.plotPath
                + self.halo.file.replace(".fits", "")
                + "_walkers"
                + self.filename_append
                + ".pdf"
            )
            plt.clf()
            plt.close(fig)
        else:
            plt.show()

    def set_labels_and_units(self):
        self.samples_units = self.samples.copy()
        samples_units = self.samples.copy()
        samples_list = list()

        x0 = np.percentile(self.samples.real[:, 1], [16, 50, 84])[1] + self.halo.fov_info_mcmc[2]
        y0 = np.percentile(self.samples.real[:, 2], [16, 50, 84])[1] + self.halo.fov_info_mcmc[0]
        
        self.centre_pix = np.array([x0, y0], dtype=np.int64)
        self.centre_wcs = wcs.utils.pixel_to_skycoord(self.centre_pix[0], self.centre_pix[1], wcs.WCS(self.halo.header), origin=1)



        samples_list = np.asarray([samples_units[:, i]for i 
                                   in range(self.dim)])

        transformed = utils.transform_units(self, np.copy(samples_list))
        for i in range(self.dim):
            self.samples_units[:, i] = transformed[i]

        self.popt_units = utils.transform_units(self, np.copy(self.popt))
        self.percentiles_units = self.get_percentiles(self.samples_units)
        self.params_units = utils.add_parameter_labels(
            self, self.percentiles_units[:, 1].reshape(self.dim)
        )
        
        self.get_units()
        uncertainties1 = self.percentiles_units[:, 1] - self.percentiles_units[:, 0]
        uncertainties2 = self.percentiles_units[:, 2] - self.percentiles_units[:, 1]
        '''
        string_to_print = "\n Parameters: \n%s \nOne sigma parameter uncertainties (lower, upper): \
                                    \n%s \n%s \nIn Units: %s" % (
                str(self.params_units[self.params]),
                str(uncertainties1),
                str(uncertainties2),
                str(self.units),
            )
        '''                         
        string_to_print = f"\n Parameters: \n{str(self.params_units[self.params])} \
            \nOne sigma parameter uncertainties (lower, upper): \
            \n{str(uncertainties1)} \n{str(uncertainties2)} \
            \nIn Units: {str(self.units)}" 
                                    
        self.log.log(logging.INFO, string_to_print)
        return string_to_print

    def get_units(self):
        labels = ["$I_0$", "$x_0$", "$y_0$"]
        units = ["$\\mu$Jy arcsec$^{-2}$", "deg", "deg"]
        fmt = [".2f", ".4f", ".4f"]

        if self.model_name == "skewed":
            labels.extend(("$r_{x^+}$", "$r_{x^-}$", "$r_{y^+}$", "$r_{y^-}$"))
            units.extend(("kpc", "kpc", "kpc", "kpc"))
            fmt.extend((".0f", ".0f", ".0f", ".0f"))
        elif self.model_name in ["ellipse", "rotated_ellipse"]:
            labels.extend(("$r_{x}$", "$r_{y}$"))
            units.extend(("kpc", "kpc"))
            fmt.extend((".1f", ".1f"))
        elif self.model_name == "circle":
            labels.append("$r_{e}$")
            units.append("kpc")
            fmt.append(".1f")
        if self.model_name in ["rotated_ellipse", "skewed"]:
            labels.append("$\\phi_e$")
            units.append("Rad")
            fmt.append(".3f")
        if self.k_exponent:
            labels.append("$k$")
            units.append(" ")
            fmt.append(".3f")
        if self.offset:
            labels.append("$C$")
            units.append(" ")
            fmt.append(".3f")

        self.labels = np.array(labels, dtype="<U30")
        self.units = np.array(units, dtype="<U30")
        self.fmt = np.array(fmt, dtype="<U30")

        self.labels_units = np.copy(self.labels)
        for i in range(self.dim):
            self.labels_units[i] = self.labels[i] + " [" + self.units[i] + "]"

    def get_confidence_interval(self, percentage=95, units=True):
        alpha = 1.0 - percentage / 100.0
        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
        se = np.zeros(self.params.shape)

        if units:
            for i in range(self.dim):
                se[self.params] = np.sqrt(
                    np.mean(self.samples_units[:, i] ** 2.0)
                    - np.mean(self.samples_units[:, i]) ** 2.0
                )
            conf_low = self.params_units - z_alpha * se
            conf_up = self.params_units + z_alpha * se
            for i in range(self.dim):
                self.log.log(
                    logging.INFO,
                    "{}% Confidence interval of {}: ({:.5f}, {:.5f}) {}".format(
                        percentage,
                        self.labels[i],
                        conf_low[i],
                        conf_up[i],
                        self.units[i],
                    ),
                )
            self.log.log(logging.INFO, "")
        else:
            for i in range(self.dim):
                se[i] = np.sqrt(
                    np.mean(self.samples[:, i] ** 2.0)
                    - np.mean(self.samples[:, i]) ** 2.0
                )
            conf_low = self.parameters - z_alpha * se
            conf_up = self.parameters + z_alpha * se
            for i in range(self.dim):
                self.log.log(
                    logging.INFO,
                    "{}% Confidence interval of {}: ({:.5f}, {:.5f})".format(
                        percentage, self.labels[i], conf_low[i], conf_up[i]
                    ),
                )
            self.log.log(logging.INFO, "")

        return [conf_low, conf_up]
    
    def set_data_to_use(self, data):
        if self.rebin:
            binned_data = utils.regridding(self.halo, data, decrease_fov=True)
            if not self.mask:
                self.image_mask = np.zeros(self.halo.data.shape)
            self.binned_image_mask = utils.regridding(
                self.halo, self.image_mask * u.Jy, mask=not self.halo.cropped
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
            
    def get_radius_estimate(self):
        if self.model_name == "circle":
            radius = - self.parameters["r1"] *np.log(3 * self.rms.value / self.parameters["I0"]) * self.halo.pix2kpc
            print(radius)
        else:
            pass

    def get_chi2_value(self, mask_treshold=0.4):
        self.mask_treshold = mask_treshold
        x = np.arange(0, self.halo.data_mcmc.shape[1], 1)
        y = np.arange(0, self.halo.data_mcmc.shape[0], 1)
        self.x_pix, self.y_pix = np.meshgrid(x, y)

        params = self.parameters.copy()
        params[1] += self.halo.margin[2]
        params[2] += self.halo.margin[0]

        binned_data = self.set_data_to_use(self.halo.data_mcmc)
        model = (
            self._func_(self, params, rotate=True).reshape(self.halo.data.shape) * u.Jy
        )
        binned_model = utils.regrid_to_beamsize(self.halo, model)

        self.rmsregrid = utils.findrms(binned_data)

        if not self.mask:
            self.image_mask = np.zeros(self.halo.data.shape)

        binned_image_mask = utils.regridding(
            self.halo, self.image_mask * u.Jy, mask=not self.halo.cropped
        ).value
        binned_model = binned_model.ravel()[
            binned_image_mask.ravel() <= mask_treshold * binned_image_mask.max()
        ]

        chi2 = np.sum(((binned_data - binned_model) / (self.rmsregrid)) ** 2.0)
        binned_dof = len(binned_data) - self.dim
        self.chi2_red = chi2 / binned_dof

        self.ln_likelihood = -np.sum(
            ((binned_data - binned_model) ** 2.0) / (2 * (self.rmsregrid) ** 2.0)
            + np.log(np.sqrt(2 * np.pi) * self.rmsregrid)
        )
        self.AIC = 2 * (self.dim - self.ln_likelihood)

        self.log.log(logging.INFO, "chi^2: {}".format(chi2))
        self.log.log(logging.INFO, "effective DoF: {}".format(binned_dof))
        self.log.log(logging.INFO, "chi^2_red: {}".format(self.chi2_red))
        # self.log.log(logging.INFO,'AIC: {}'.format(self.AIC))

        x = np.arange(0, self.data.shape[1], 1)
        y = np.arange(0, self.data.shape[0], 1)
        self.x_pix, self.y_pix = np.meshgrid(x, y)
        return self.chi2_red

    def get_flux(self, int_max=np.inf, freq=None):
        self.int_max = int_max
        if freq is None:
            freq = self.halo.freq

        a = self.samples[:, 3] * self.halo.pix_size
        if self.model_name == "skewed":
            b = self.samples[:, 5] * self.halo.pix_size
            c = self.samples[:, 4] * self.halo.pix_size
            d = self.samples[:, 6] * self.halo.pix_size
            factor = a * b + c * d + a * d + b * c

        elif self.model_name in ["ellipse", "rotated_ellipse"]:
            b = self.samples[:, 4] * self.halo.pix_size
            factor = 4 * a * b
        else:
            factor = 4 * a**2
        if self.k_exponent:
            m = self.samples[:, self.at("k_exp")] + 0.5
        else:
            m = 0.5

        I0 = u.Jy * self.samples[:, 0] / self.halo.pix_area
        flux = (
            gamma(1.0 / m)
            * np.pi
            * I0
            / (4 * m)
            * factor
            * gammainc(1.0 / m, int_max ** (2 * m))
            * (freq / self.halo.freq) ** self.alpha
        ).to(u.mJy)

        self.flux = np.copy(flux)
        self.flux_freq = freq
        self.flux_val = np.percentile(flux, 50)
        self.flux_err = (np.percentile(flux, 84) - np.percentile(flux, 16)) / 2.0

        # cal = 0.1
        # sub = 0.1 # Osinga et al. 2020

        # self.flux_std = np.sqrt((cal*self.flux_val.value)**2+sub**2+flux_err**2)*u.mJy
        # self.flux_err = np.sqrt((cal*self.flux.value)**2+sub**2+flux_err**2)*u.mJy
        self.log.log(
            logging.INFO,
            "MCMC Flux at {:.1f} {}: {:.2f} +/- {:.2f} {}".format(
                freq.value,
                freq.unit,
                self.flux_val.value,
                self.flux_err.value,
                flux.unit,
            ),
        )
        self.log.log(logging.INFO, "Integration radius " + str(int_max))
        self.log.log(
            logging.INFO,
            "S/N based on flux {:.2f}".format(
                self.flux_val.value / self.flux_err.value
            ),
        )
        
        return self.flux_val, self.flux_err

    def get_power(self, freq=None):
        if freq is None:
            freq = self.halo.freq

        cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        d_L = cosmology.luminosity_distance(self.halo.z)
        power = (
            4
            * np.pi
            * d_L**2.0
            * ((1.0 + self.halo.z) ** ((-1.0 * self.alpha) - 1.0))
            * self.flux
            * ((freq / self.flux_freq) ** self.alpha)
        ).to(u.W / u.Hz)
        power_std = (
            4
            * np.pi
            * d_L**2.0
            * ((1.0 + self.halo.z) ** ((-1.0 * self.alpha) - 1.0))
            * self.flux_err
            * ((freq / self.flux_freq) ** self.alpha)
        ).to(u.W / u.Hz)
        self.power_std = np.percentile(power_std, 50)

        cal = 0.1
        sub = 0.1  # Osinga et al. 2020
        self.power = np.copy(power)
        self.power_val = np.percentile(power, [50])[0]
        power_err = (
            (np.percentile(power, [84])[0] - np.percentile(power, [16])[0]) / 2.0
        ).value
        self.power_std = np.sqrt(
            (cal * self.power_val.value) ** 2 + sub**2 + power_err**2
        )
        self.log.log(
            logging.INFO,
            "Power at {:.1f} {}: ({:.3g} +/- {:.3g}) {}".format(
                freq.value,
                freq.unit,
                np.percentile(power, [50])[0].value,
                (np.percentile(power, [84])[0] - np.percentile(power, [16])[0]).value
                / 2.0,
                power.unit,
            ),
        )
