#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: J.M. Boxelaar
Version: 08 June 2020
"""

from __future__ import division
import sys
import os
import json
from logging import Logger
import emcee

import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count, freeze_support, set_start_method
from scipy import ndimage
from scipy.optimize import curve_fit
from skimage.measure import block_reduce
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits


# Subfile imports
from . import plot_fits
from . import fdca_utils as utils
from .halo_object import RadioHalo
from .processing import Processing

if __name__ == "__main__":
    try:
        set_start_method("fork")
        freeze_support()
    except RuntimeError as e:
        print("multiprocess error:",e)

rad2deg = 180.0 / np.pi
deg2rad = np.pi / 180.0
Jydeg2 = u.Jy / (u.deg * u.deg)
mJyarcsec2 = u.mJy / (u.arcsec * u.arcsec)
uJyarcsec2 = 1.0e-3 * u.mJy / (u.arcsec * u.arcsec)
max_params = 10

class BaseFitting():
    def __init__(
        self,
        _parent_: RadioHalo,
        data = None,
        model: str = "circle",
        walkers: int = 100,
        steps: int = 1200,
        burntime = None,
        logger = None,
        rebin: bool = True,
        max_radius = None,
        freeze_params: dict = {},
        gamma_prior: bool = False,
        k_exponent: bool = False,
        offset: bool = False,
        debug: bool = True,
        num_cpus: int = -1,
    ):
        assert model in ["circle", "ellipse", "rotated_ellipse", "skewed"], "Provide valid function kind"
        
        self.data = _parent_.data_mcmc if data is None else data
        self.logger: Logger = _parent_.logger if logger is None else logger
        if debug: self.logger.info(f"Initialising model: {model}")

        self.orig_shape = _parent_.data.shape
        self.rebin = rebin
        self.halo = _parent_
        self.noise = _parent_.imagenoise
        self.mask = _parent_.mask
        self.rms = _parent_.rmsnoise
        self.sigma = (self.rms * self.halo.beam2pix).value
        self.steps = int(steps)
        self.mask_treshold = 0.5
        self.k_exponent = k_exponent
        self.offset = offset
        self.gamma_prior = gamma_prior
        self.max_cpu = num_cpus

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
            self.logger.error("Invalid model name")
            sys.exit()
            
        self.AppliedParameters[-2] = True if self.k_exponent else False
        self.AppliedParameters[-1] = True if self.offset else False
        
        for freeze, freez_vals in freeze_params.copy().items():
            if freeze not in self.paramNames:
                self.logger.error(f"Parameter {freeze} not a model parameters\nChoose from: {', '.join(self.paramNames)}")
                freeze_params.pop(freeze)
        
        applied_frozen = [False] * len(self.AppliedParameters)
        self.frozen_vals = np.asarray(list(freeze_params.values()))
        freeze_keys = np.asarray(list(freeze_params.keys()))
        for i, param in enumerate(self.paramNames):
            if param in freeze_keys:
                applied_frozen[i] = True
                if param == "x0":
                    self.frozen_vals[freeze_keys==param] -= _parent_.fov_info_mcmc[2]
                if param == "y0":
                    self.frozen_vals[freeze_keys==param] -= _parent_.fov_info_mcmc[0]
        
        self.frozen = pd.DataFrame.from_dict(
            {"frozen": applied_frozen}, 
            orient = "index", 
            columns = self.paramNames
        ).loc["frozen"]

        self.params = pd.DataFrame.from_dict(
            {"params": self.AppliedParameters}, 
            orient = "index", 
            columns = self.paramNames
        ).loc["params"]
        
        self.dim = len(self.params[self.params & ~self.frozen])
        self.walkers = int(walkers) if (walkers >= 2 * self.dim) else int(2 * self.dim + 4)

        if burntime is None:
            self.burntime = int(0.125 * self.steps)
        elif burntime < 1 or burntime >= 0.8 * self.steps:
            self.logger.warning(f"MCMC Input burntime of {burntime} is invalid. setting burntime to {int(0.25*self.steps)}")
            self.burntime = int(0.25 * self.steps)
        else:
            self.burntime = int(burntime)
        
        if max_radius is None:
            self.max_radius = self.data.shape[0] / 2.0
        else:
            self.max_radius = max_radius / self.halo.pix2kpc.value
        
        if debug:
            str_to_log = f"Applied parameters: {', '.join(np.asarray(self.paramNames)[np.asarray(self.AppliedParameters)])}"
            if np.any(self.frozen):
                str_to_log += f"  (Freezing parameters: {', '.join(freeze_keys)} to: {', '.join(map(str, self.frozen_vals))})"
            else:
                str_to_log += "  (No parameters frozen)"
            self.logger.info(str_to_log)
            self.logger.debug("Number of walkers set to %d" % self.walkers)
            self.logger.debug("Burntime set to %d" % self.burntime)
            self.logger.debug("Maximum radius set to %f pix" % self.max_radius)
                
        x = np.arange(0, self.data.shape[1], 1)
        y = np.arange(0, self.data.shape[0], 1)
        self.x_pix, self.y_pix = np.meshgrid(x, y)
        
        # define the mask based on user ds9 region
        image_mask = utils.masking(self, full_size=True)
        if np.any(np.asarray(self.halo.fov_info_mcmc) < 0):
            image_mask, __ = utils.pad_image(image_mask.astype(int))
            self.image_mask = image_mask.astype(bool)
        else:
            self.image_mask = image_mask[
                self.halo.fov_info_mcmc[0] : self.halo.fov_info_mcmc[1],
                self.halo.fov_info_mcmc[2] : self.halo.fov_info_mcmc[3],
            ]
            
        self.data_to_use = utils.set_data_to_use(self, self.data)
        self.mcmc_noise = utils.findrms(self.data_to_use)
        
        # need to set binned image data to be used in model_data_to_use()
        self.binned_image_mask = utils.regridding(
            self.halo, 
            self.image_mask.astype(int) * u.Jy, 
            decrease_fov=self.halo.cropped, 
            mask=self.mask
        ).value
        
        
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
            fmt.append(".3e")
        if self.offset:
            labels.append("$C$")
            units.append(" ")
            fmt.append(".3f")

        self.labels = np.array(labels, dtype="<U30")
        self.units = np.array(units, dtype="<U30")
        self.fmt = np.array(fmt, dtype="<U30")

        self.labels_units = np.copy(self.labels)
        for i in range(len(self.labels)):
            self.labels_units[i] = self.labels[i] + " [" + self.units[i] + "]"
                


class SingleComponentFitting(BaseFitting):
    def __init__(self, _parent_: RadioHalo, p0=None, bounds=None, **kwargs):
        BaseFitting.__init__(self, _parent_, **kwargs)
        
        p0_temp, bounds_temp = utils.get_initial_guess(_parent_)
        self.p0 = p0_temp if p0 is None else p0
        self.bounds = bounds_temp if bounds is None else bounds
        
        filename_append = "_%s" % (self.model_name)
        if self.mask:
            filename_append += "_mask"
        if self.k_exponent:
            filename_append += "_exp"
        if self.offset:
            filename_append += "_offset"
        if np.any(self.frozen):
            filename_append += "_frozen-" + "-".join(list(self.frozen[self.frozen].keys()))
        self.filename_append = filename_append


    def __pre_fit(self):
        popt, perr = self.pre_mcmc_fit(
            self.data, p0=np.array(self.p0), bounds=np.array(self.bounds)
        )
        return popt, perr


    def run(self, pre_fit_guess=None, save=False, save_path=""):
        coord = np.meshgrid(
            np.arange(0, self.data.shape[1]), 
            np.arange(0, self.data.shape[0])
        )
        
        if pre_fit_guess is None:
            if not hasattr(self, "popt"):
                self.logger.info("Prepare MCMC fitting...")
                self.popt = self.__pre_fit()[0]
        else:
            self.popt = pre_fit_guess
        
        pos = list()
        for _ in range(self.walkers):
            popt = self.popt[self.params & ~self.frozen] * (1.0 + 1.0e-3 * np.random.randn(self.dim))
            pos.append(popt)
        
        self.halo_info = set_dictionary(self)

        
        num_CPU = cpu_count() if self.max_cpu == -1 else self.max_cpu
        if self.max_cpu > cpu_count():
            self.logger.warning(f"Number of CPU's requested ({self.max_cpu}) is higher than available ({cpu_count()})")
            num_CPU = cpu_count()
        self.logger.info(f"Starting MCMC run (number of CPU's: {num_CPU})...")
        
        if num_CPU == 1:
            sampler = emcee.EnsembleSampler(
                self.walkers, 
                self.dim, 
                lnprob, 
                args=[self.data_to_use, coord, self.halo_info]
            )
            sampler.run_mcmc(pos, self.steps, progress=True)
        else:
            with Pool(num_CPU) as pool:
                sampler = emcee.EnsembleSampler(
                    self.walkers, 
                    self.dim, 
                    lnprob, 
                    pool=pool, 
                    args=[self.data_to_use, coord, self.halo_info]
                )
                sampler.run_mcmc(pos, self.steps, progress=True)

        self.sampler = sampler.chain
        self.samples = self.sampler[:,int(self.burntime):,:].reshape((-1, self.dim))
        self.info = self.set_sampler_header(fits.PrimaryHDU().header)

        if save:
            self.save(save_path)
            self.get_units()
            plot_fits.samplerplot(self)
            plot_fits.cornerplot(self)

        return self.sampler

    def save(self, path:str = ""):
        if path == "":
            path = "%s%s_mcmc_samples%s.fits" % (
                self.halo.modelPath,
                self.halo.file.replace(".fits", ""),
                self.filename_append,
            )
        self.logger.debug("Saving MCMC samples to %s" % path)
        self.save_json(path)
        return self
        
    def load(self, path:str=""):
        if path == "":
            path = "%s%s_mcmc_samples%s.fits" % (
                self.halo.modelPath,
                self.halo.file.replace(".fits", ""),
                self.filename_append,
            )
        self.logger.debug("Loading MCMC samples from %s" % path)
        self.load_json(path)
        return self

    def at(self, parameter):
        par = np.array(self.paramNames)[self.params]
        return np.where(par == parameter)[0][0]
    
    def pre_mcmc_func(self, obj, *theta):
        theta = utils.add_labels(obj, theta)
        model = self._func_(obj, theta)
        return model[obj.image_mask.ravel()]


    def pre_mcmc_fit(self, image: np.ndarray, p0, bounds):
        data = image.ravel()
        data = data[self.image_mask.ravel()]

        bounds = (list(bounds[0, self.params & ~self.frozen]), list(bounds[1, self.params & ~self.frozen]))
        popt, pcov = curve_fit(
            self.pre_mcmc_func, self, data, p0=tuple(p0[self.params & ~self.frozen]), bounds=bounds
        )
        perr = np.sqrt(np.diag(pcov))
        popt = utils.add_labels(self, popt)
        perr = utils.add_labels(self, perr)

        #popt[self.frozen] = self.frozen_vals

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
        
        
        popt_units = utils.transform_units(self, popt.copy())
        
        self.logger.debug(
            "MCMC initial guess: \n{} \n and units: muJy/arcsec2, deg, deg, r_e: kpc, rad".format(
                popt_units, perr
            )
        )

        x = np.arange(0, self.data.shape[1], 1)
        y = np.arange(0, self.data.shape[0], 1)
        self.x_pix, self.y_pix = np.meshgrid(x, y)
        return popt, perr
    
    def save_json(self, path:str = ""):
        halo_info = self.halo_info
        info = dict()
        info['object'] = self.halo.name
        info['filename'] = self.halo.path
        info['maskPath'] = self.halo.maskPath
        info['outputPath'] = self.halo.basedir
        info['redshift'] = self.halo.z
        
        for key, value in halo_info.items():
            if key == "image_mask":
                continue
            if key == "binned_image_mask":
                continue
            
            if isinstance(value, u.quantity.Quantity):
                if str(value.unit) == "":
                    info[key] = value.value
                else:
                    info[key] = {"value": value.value, "unit": str(value.unit)}
            elif isinstance(value, np.float32):
                info[key] = float(value)
            elif isinstance(value, np.ndarray):
                info[key] = value.tolist()
            elif callable(value):
                info[key] = str(value.__name__)
            elif isinstance(value, pd.Series):
                info[key] = value.to_dict()
            elif isinstance(value, wcs.WCS):
                continue
            else:
                info[key] = value
                
        info["p0"] = np.asarray(self.p0, dtype=float).tolist()
        info["filename_append"] = self.filename_append
        info["walkers"] = self.walkers
        info["steps"] = self.steps
        info["burntime"] = self.burntime
        #info["bounds"] = self.bounds
        
        best = dict()
        all_units = np.asarray(["JY/PIX", "PIX", "PIX", "PIX", "PIX", "PIX", "PIX", "RAD", "NONE", "NONE"])
        for i in range(len(self.popt[self.params])):
            best[self.params.keys()[self.params][i]] = {"value": self.popt[self.params][i], "unit": all_units[self.params.values][i]}
        
        info["initial"] = best
        info["data"] = self.sampler.tolist()
        
        with open(path.replace(".fits", ".json"), "w") as f:
            json.dump(info, f, indent=4)

    def load_json(self, path:str = ""):
        with open(path.replace(".fits", ".json"), "r") as f:
            info = json.load(f)
            
        self.walkers = info["walkers"]
        self.steps = info["steps"]
        self.dim = info["dim"]
        self.burntime = info["burntime"]
        self.halo.name = info["object"]
        self.halo.path = info["filename"]
        #info['maskPath'] = self.halo.maskPath
        self.mask = info["mask"]
        self.filename_append = info["filename_append"]
        self.p0 = np.asarray(info["p0"])
        
        self.sampler = np.asarray(info["data"])
        self.samples = self.sampler[:, int(self.burntime):].reshape(
            (-1, self.dim)
        )
        
        self.params = pd.DataFrame.from_dict(
            {"params": info["params"].values()}, 
            orient = "index", 
            columns = info["params"].keys()
        ).loc["params"]
        
        self.frozen = pd.DataFrame.from_dict(
            {"frozen": info["frozen"].values()}, 
            orient = "index", 
            columns = info["frozen"].keys()
        ).loc["frozen"]
        
        popt = []
        for i in range(len(self.params)):
            if self.params[i]:
                popt.append(info["initial"][self.params.keys()[i]]["value"])
        self.popt = utils.add_labels(self, np.asarray(popt))
        


    def set_sampler_header(self, header:fits.Header):
        header["nwalkers"] = self.walkers
        header["steps"] = self.steps
        header["dim"] = self.dim
        header["burntime"] = self.burntime
        header["OBJECT"] = (self.halo.name, "Object which was fitted")
        header["IMAGE"] = self.halo.file

        all_units = np.asarray(["JY/PIX", "PIX", "PIX", "PIX", "PIX", "PIX", "PIX", "RAD", "NONE", "NONE"])
        for i in range(len(self.popt[self.params])):
            header["INIT_" + str(i)] = (
                self.popt[self.params][i],
                "MCMC initial guess",
            )
            header["UNIT_" + str(i)] = (all_units[self.params.values][i], "unit of fit parameter")
                
        header["MASK"] = (self.mask, "was the data masked during fitting")
        return header
        
    def get_sampler_header(self, header:fits.Header):
        self.walkers = header["nwalkers"]
        self.steps = header["steps"]
        self.dim = header["dim"]
        self.burntime = header["burntime"]
        self.halo.name = header["OBJECT"]
        self.halo.file = header["IMAGE"]
        
        popt = np.zeros(len(self.p0))
        for i in range(len(popt[self.params])):
            if self.params[i]:
                popt[i] = header["INIT_" + str(i)]

        self.popt = popt
        self.mask = header["MASK"]
        
    def model(self, theta, rotate=False, regrid=False):
        theta = utils.add_parameter_labels(self, theta)
        if not regrid:
            return self._func_(self, theta, rotate)
        else:
            info = set_dictionary(self)
            coords = np.meshgrid(
                np.arange(0, self.data.shape[1]), 
                np.arange(0, self.data.shape[0])
            )
            model = self._func_mcmc(info, coords, theta, rotate=True)
            return set_model_to_use(info, model)
        
    def get_samples(self) -> np.ndarray:
        return self.samples.T
    
    def get_param_names(self) -> list:
        return list(self.params[self.params & ~self.frozen].keys())
    
    @property    
    def results(self):
        if not hasattr(self, "processing_object"):
            self.logger.debug("Initialising Processing object")
            self.processing_object = Processing(self, save=True)
        return self.processing_object
        
    def get_results(self):
        return self.results
    
    def clear_results(self):
        del self.processing_object
        


def set_dictionary(obj: BaseFitting) -> dict:
    return {
        "model_name": obj.model_name,
        "dim": obj.dim,
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
        "params": obj.params,
        "paramNames": obj.paramNames,
        "gamma_prior": obj.gamma_prior,
        "fov_info_mcmc": obj.halo.fov_info_mcmc,
        "wcs": obj.halo.wcs,
        "cropped": obj.halo.cropped,
        "frozen": obj.frozen,
        "frozen_vals": obj.frozen_vals
    }


def set_model_to_use(info, array):
    binned_data = regrid_to_beamsize(info, array.value)
    return binned_data.ravel()[
        info["binned_image_mask"].ravel()
        >= info["mask_treshold"] * info["binned_image_mask"].max()
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
        theta["x0"],
        theta["y0"],
        theta["r1"],
        theta["r3"],
        theta["ang"],
        theta["k_exp"],
        1.0,
        1.0,
    )
    G_mm = G(
        x,
        y,
        theta["x0"],
        theta["y0"],
        theta["r2"],
        theta["r4"],
        theta["ang"],
        theta["k_exp"],
        -1.0,
        -1.0,
    )
    G_pm = G(
        x,
        y,
        theta["x0"],
        theta["y0"],
        theta["r1"],
        theta["r4"],
        theta["ang"],
        theta["k_exp"],
        1.0,
        -1.0,
    )
    G_mp = G(
        x,
        y,
        theta["x0"],
        theta["y0"],
        theta["r2"],
        theta["r3"],
        theta["ang"],
        theta["k_exp"],
        -1.0,
        1.0,
    )
    Ir = theta["I0"] * (G_pp + G_pm + G_mm + G_mp) + theta["off"]
    return convolve_with_gaussian(info, Ir, rotate)


def G(x, y, x0, y0, re_x, re_y, ang, k_exp, sign_x, sign_y):
    x_rot = (x - x0) * np.cos(ang) + (y - y0) * np.sin(ang)
    y_rot = -(x - x0) * np.sin(ang) + (y - y0) * np.cos(ang)
    func = (np.sqrt(sign_x * x_rot) ** 4.0) / (re_x**2.0) + (
        np.sqrt(sign_y * y_rot) ** 4.0
    ) / (re_y**2.0)

    exponent = np.exp(- func ** (0.5 + k_exp))
    exponent[np.where(np.isnan(exponent))] = 0.0
    return exponent

def lnL_multicomponent(theta, data, coord, full_info):
    kwargs = {"rotate": True}
    model = np.zeros(data.shape)
    for i, info in enumerate(full_info):
        raw_model = info["_func_"](info, coord, theta[i], **kwargs) * u.Jy
        model += set_model_to_use(info, raw_model)
    return - 0.5 * np.sum( ((data - model) / info["sigma"]) ** 2 + np.log(2 * np.pi * (info["sigma"]) ** 2)) 

def lnL(theta, data, coord, info):
    kwargs = {"rotate": True}
    raw_model = info["_func_"](info, coord, theta, **kwargs) * u.Jy
    model = set_model_to_use(info, raw_model)
    return - 0.5 * np.sum( ((data - model) / info["sigma"]) ** 2 + np.log(2 * np.pi * (info["sigma"]) ** 2)) 

def lnprior(theta, shape, info):
    prior = -np.inf
    if (theta["I0"] > 0) and (-0.4 < theta["k_exp"] < 19):
        if (0 <= theta["x0"] < shape[1]) and (0 <= theta["y0"] < shape[0]):
            if 0 < theta["r1"] < info["max_radius"]:
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
    ptheta = add_parameter_labels(info["params"], info["paramNames"])
    ptheta[info["params"] & ~info["frozen"]] = theta
    ptheta[info["frozen"]] = info["frozen_vals"]

    if info["model_name"] == "skewed":
        lp = lnprior8(ptheta, coord[0].shape, info)
    else:
        lp = lnprior(ptheta, coord[0].shape, info)
    if not np.isfinite(lp):
        return -np.inf
    
    #likelihood = lnL(ptheta, data, coord, info) + lp
    return lnL(ptheta, data, coord, info) + lp


def lnprob_multicomponent(full_theta, data, coord, full_info):
    full_ptheta = list()
    idx = 0
    lp = 0.0
    for i, info in enumerate(full_info):
        
        ptheta = add_parameter_labels(info["params"], info["paramNames"])
        ptheta[info["params"] & ~info["frozen"]] = full_theta[idx:idx+info["dim"]]
        ptheta[info["frozen"]] = info["frozen_vals"]
        if info["linked_loc"] and i > 0:
            ptheta['x0'] = full_ptheta[0]['x0']
            ptheta['y0'] = full_ptheta[0]['y0']
        if i == 0:
            I0 = ptheta["I0"]
        elif ptheta["I0"] > I0:
            return -np.inf
        else:
            I0 = ptheta["I0"]
        
        if info["model_name"] == "skewed":
            lp += lnprior8(ptheta, coord[0].shape, info)
        else:
            lp += lnprior(ptheta, coord[0].shape, info)

        if not np.isfinite(lp):
            return -np.inf
        
        full_ptheta.append(ptheta)
        idx += (i+1) * info["dim"]
    
    likelihood = lnL_multicomponent(full_ptheta, data, coord, full_info) + lp
    return likelihood


def lnprob_multi(theta, data, coord, info):
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


def add_parameter_labels(params, paramNames, array=None):
    full_array = np.zeros(params.shape)
    array = np.ones(params[params].shape) if array is None else array
    
    full_array[params == True] = array
    parameterised_array = pd.DataFrame.from_dict(
        {"params": full_array}, orient="index", columns=paramNames
    ).loc["params"]
    return parameterised_array



class MultiComponentFitting(BaseFitting):
    def __init__(
        self,
        _parent_: RadioHalo,
        p0: list[list] = None,
        bounds: list[list] = None,
        model: list[str] = ["circle", "rotated_ellipse"],
        link_loc: list[bool] = [False, False], # not implemented yet
        profiles: list[str] = ["default", "default"],
        **kwargs
    ):
        BaseFitting.__init__(self, _parent_, model=model[0], debug=False, **kwargs)
        assert len(link_loc) == len(model), "link_loc must have the same length as the number of components"
        assert len(profiles) == len(model), "exponent_shape must have the same length as the number of components"
        
        fits: list[BaseFitting] = list()
    
        for i, component in enumerate(model):
            freeze_params = {}
            if link_loc[i] and i > 0:
                link_loc[0] = True
                freeze_params.update({"x0": -1, "y0": -1})
            if profiles[i] == "gaussian":
                kwargs.update({"k_exponent": True})
                freeze_params.update({"k_exp": 0.5})
            elif profiles[i] == "free":
                kwargs.update({"k_exponent": True})
            
            kwargs.update({"freeze_params": freeze_params})
            fits.append(BaseFitting(_parent_, model=component, **kwargs))
        self.link = np.asarray(link_loc, dtype=bool)
        
        p0_temp, bounds_temp = utils.get_initial_guess(_parent_)
        if p0 is None:
            self.p0 = np.asarray([p0_temp for _ in range(len(fits))])
        else:
            assert len(p0) == len(fits), "p0 must have the same length as the number of components"
            self.p0 = np.asarray(p0)
        
        if bounds is None:
            self.bounds = np.asarray([bounds_temp for _ in range(len(fits))])
        else:
            assert len(bounds) == len(fits), "bounds must have the same length as the number of components"
            self.bounds = np.asarray(bounds)
            
        for fit in fits[1:]:
            self.dim += fit.dim
            self.AppliedParameters = np.concatenate((self.AppliedParameters, fit.AppliedParameters)).reshape(2,-1)
            self.paramNames = np.concatenate((self.paramNames, fit.paramNames)).reshape(2,-1)
        
        if self.walkers <= 2*self.dim:
            self.logger.debug("Number of walkers is less than 2*dim, increasing walkers to 2 * dim + 1")
            self.walkers = 2*self.dim + 1
            for fit in fits:
                fit.walkers = self.walkers
            
        self.params = pd.concat([fit.params for fit in fits], axis=1)
        self.params.columns = [f"comp_{i}" for i in range(len(fits))]
        self.prms = self.params.values.T
        
        self.frozen = pd.concat([fit.frozen for fit in fits], axis=1)
        self.frozen.columns = [f"comp_{i}" for i in range(len(fits))]
        self.frzn = self.frozen.values.T
        
        filename_append = "multicomponent"
        if self.mask:
            filename_append += "_mask"
        if self.k_exponent:
            filename_append += "_exp"
        if self.offset:
            filename_append += "_offset"
        if np.any(self.frozen):
            filename_append += "_frozen-" + "-".join(list(self.frozen[self.frozen].keys()))
        self.filename_append = filename_append + f"_{len(model):02d}components"
        
        self.fits = fits
        
    def run(self, pre_fit_guess=None, save=False, save_path=""):
        coord = np.meshgrid(
            np.arange(0, self.data.shape[1]), 
            np.arange(0, self.data.shape[0])
        )
        
        if np.any(self.link):
            for i, fit in enumerate(self.fits):
                if i ==0:
                    self.p0[0][fit.frozen] = fit.frozen_vals
                elif self.link[i]:
                    fit.frozen_vals[fit.frozen[fit.frozen].index.get_loc("x0")] = self.p0[0][fit.frozen.index.get_loc("x0")]
                    fit.frozen_vals[fit.frozen[fit.frozen].index.get_loc("y0")] = self.p0[0][fit.frozen.index.get_loc("y0")]

        if pre_fit_guess is None:
            if not hasattr(self, "popt"):
                self.logger.info("Prepare MCMC fitting...")
                popt = self.__pre_fit()[0]
                popt = utils.set_linked_loc(self, popt)
                self.popt = popt.values.T
        else:
            self.popt = pre_fit_guess
        
        pos = list()
        for _ in range(self.walkers):
            popt = self.popt[self.prms & ~self.frzn] * (1.0 + 1.0e-3 * np.random.randn(self.dim))
            pos.append(popt)
        
        self.halo_info_list = list()
        for i, fit in enumerate(self.fits):
            dictionary = set_dictionary(fit)
            dictionary["linked_loc"] = self.link[i]
            self.halo_info_list.append(dictionary)
        

        num_CPU = cpu_count() if self.max_cpu == -1 else self.max_cpu
        if self.max_cpu > cpu_count():
            self.logger.warning(f"Number of CPU's requested ({self.max_cpu}) is higher than available ({cpu_count()})")
            num_CPU = cpu_count()
        self.logger.info(f"Starting MCMC run (number of CPU's: {num_CPU})...")
        if num_CPU == 1:
            sampler = emcee.EnsembleSampler(
                self.walkers, 
                self.dim, 
                lnprob_multicomponent,
                args=[self.data_to_use, coord, self.halo_info_list]
            )
            sampler.run_mcmc(pos, self.steps, progress=True, skip_initial_state_check = True)
        else:
            with Pool(num_CPU) as pool:
                sampler = emcee.EnsembleSampler(
                    self.walkers, 
                    self.dim, 
                    lnprob_multicomponent, 
                    pool=pool, 
                    args=[self.data_to_use, coord, self.halo_info_list]
                )
                sampler.run_mcmc(pos, self.steps, progress=True, skip_initial_state_check = True)

        self.sampler = sampler.chain
        self.samples = self.sampler[:, int(self.burntime):, :].reshape(
            (-1, self.dim)
        )

        if save:
            self.save(save_path)
            for i, fit in enumerate(self.fits):
                fit.get_units()
                if i == 0:
                    self.units = fit.units
                    self.labels = np.asarray([label + f"_c{i}" for label in fit.labels])
                    self.fmt = fit.fmt
                else:
                    self.units = np.concatenate((self.units, fit.units))
                    self.labels = np.concatenate((self.labels, np.asarray([label + f"_c{i}" for label in fit.labels])))
                    self.fmt = np.concatenate((self.fmt, fit.fmt))
                
            labels_units = list()
            for i in range(len(self.labels)):
                labels_units.append(self.labels[i] + " [" + self.units[i] + "]")
            self.labels_units = np.asarray(labels_units)
                
            plot_fits.samplerplot(self)
            plot_fits.cornerplot(self)

        return self.sampler
    
    def save(self, path:str = ""):
        if path == "":
            path = "%s%s_mcmc_samples%s.fits" % (
                self.halo.modelPath,
                self.halo.file.replace(".fits", ""),
                self.filename_append,
            )
        self.logger.debug("Saving MCMC samples to %s" % path)
        self.save_json(path)
        return self
        
    def load(self, path:str=""):
        if path == "":
            path = "%s%s_mcmc_samples%s.fits" % (
                self.halo.modelPath,
                self.halo.file.replace(".fits", ""),
                self.filename_append,
            )
        self.logger.debug("Loading MCMC samples from %s" % path)
        self.load_json(path)
        return self
    
    def __pre_fit(self):
        popt, perr = self.pre_mcmc_fit(
            self.data, p0=np.array(self.p0), bounds=np.array(self.bounds)
        )
        return popt, perr

    
    def pre_mcmc_fit(self, image: np.ndarray, p0, bounds):
        data = image.ravel()
        data = data[self.image_mask.ravel()]
        
        bounds = (
            list(self.bounds[0,0,self.prms[0] & ~self.frzn[0]]) + list(self.bounds[1,0,self.prms[1] & ~self.frzn[1]]),
            list(self.bounds[0,1,self.prms[0] & ~self.frzn[0]]) + list(self.bounds[1,1,self.prms[1] & ~self.frzn[1]])
        )

        full_popt, pcov = curve_fit(
            pre_mcmc_func, self, data, p0=tuple(p0[self.prms & ~self.frzn]), bounds=bounds
        )
        full_perr = np.sqrt(np.diag(pcov))
        
        #full_popt = np.ones(self.dim)
        #full_perr = np.zeros(self.dim)

        final_popt = self.params.copy()
        final_perr = self.params.copy()

        idx = 0
        for i, fit in enumerate(self.fits):
            popt = utils.add_labels(fit, full_popt[idx:idx+fit.dim])
            perr = utils.add_labels(fit, full_perr[idx:idx+fit.dim])

            if not self.k_exponent:
                popt["k_exp"] = 0.5
            if not self.offset:
                popt["off"] = 0.0

            if i == 0:
                I0 = popt["I0"]
            elif popt["I0"] > I0:
                popt["I0"] = I0
            else:
                I0 = popt["I0"]

            if fit.model_name == "skewed":
                """longest dimension of elliptical shape should always be the x-axis.
                This routine switches x and y if necessary to accomplish this."""
                if (popt["r1"] + popt["r2"]) <= (
                    popt["r3"] + popt["r4"]
                ):
                    popt["r1"], popt["r3"] = popt["r3"], popt["r1"]
                    popt["r2"], popt["r4"] = popt["r4"], popt["r3"]
                    popt["ang"] += np.pi / 2.0

            if fit.model_name in ["ellipse", "rotated_ellipse"]:
                if popt["r1"] <= popt["r2"]:
                    popt["r1"], popt["r2"] = popt["r2"], popt["r1"]
                    popt["ang"] += np.pi / 2.0

            if fit.model_name in ["rotated_ellipse", "skewed"]:
                """Angle of ellipse from positive x should be between 0 and pi."""
                popt["ang"] = popt["ang"] % (2 * np.pi)
                if popt["ang"] >= np.pi:
                    popt["ang"] -= np.pi

            for r in range(4):
                r += 1
                if popt["r" + str(r)] > self.max_radius:
                    popt["r" + str(r)] = self.max_radius
            
            popt_units = utils.transform_units(self, popt.copy())
            
            self.logger.debug(
                "MCMC initial guess Component {:02d}: \n{} \n and units: muJy/arcsec2, deg, deg, r_e: kpc, rad".format(i, popt_units)
            )
            final_popt[f"comp_{i}"] = popt
            final_perr[f"comp_{i}"] = perr
            idx += (i+1) * fit.dim 
        return final_popt, final_perr
    
    def set_sampler_header(self, header:fits.Header):
        header["nwalkers"] = self.walkers
        header["steps"] = self.steps
        header["dim"] = self.dim
        header["burntime"] = self.burntime
        header["OBJECT"] = (self.halo.name, "Object which was fitted")
        header["IMAGE"] = self.halo.file

        all_units = np.asarray(["JY/PIX", "PIX", "PIX", "PIX", "PIX", "PIX", "PIX", "RAD", "NONE", "NONE"])
        all_units = np.concatenate([all_units for _ in range(len(self.fits))]).reshape(len(self.fits),-1)
        for i in range(len(self.popt[self.prms])):
            header["INIT_" + str(i)] = (
                self.popt[self.prms][i],
                "MCMC initial guess",
            )
            header["UNIT_" + str(i)] = (all_units[self.prms][i], "unit of fit parameter")
                
        header["MASK"] = (self.mask, "was the data masked during fitting")
        return header
    
    def get_sampler_header(self, header:fits.Header):
        self.walkers = header["nwalkers"]
        self.steps = header["steps"]
        self.dim = header["dim"]
        self.burntime = header["burntime"]
        self.halo.name = header["OBJECT"]
        self.halo.file = header["IMAGE"]
        
        popt = np.zeros_like(self.p0).ravel()
        j = 0
        for i in range(len(self.prms.ravel())):
            if self.prms.ravel()[i]:
                popt[i] = header["INIT_" + str(j)]
                j += 1

        self.popt = popt.reshape(self.p0.shape)
        self.mask = header["MASK"]
        
    def save_json(self, path:str = ""):
        info = dict()
        info['object'] = self.halo.name
        info['filename'] = self.halo.path
        info['maskPath'] = self.halo.maskPath
        info['outputPath'] = self.halo.basedir
        info['redshift'] = self.halo.z
        
        
        for i, halo_info in enumerate(self.halo_info_list):
            comp_dict = dict()
            for key, value in halo_info.items():
                if key == "image_mask":
                    continue
                if key == "binned_image_mask":
                    continue
            
                if isinstance(value, u.quantity.Quantity):
                    if str(value.unit) == "":
                        comp_dict[key] = value.value
                    else:
                        comp_dict[key] = {"value": value.value, "unit": str(value.unit)}
                elif isinstance(value, np.float32):
                    comp_dict[key] = float(value)
                elif isinstance(value, np.ndarray):
                    comp_dict[key] = value.tolist()
                elif callable(value):
                    comp_dict[key] = str(value.__name__)
                elif isinstance(value, pd.Series):
                    comp_dict[key] = value.to_dict()
                elif isinstance(value, wcs.WCS):
                    continue
                elif key == "linked_loc":
                    comp_dict[key] = bool(value)
                else:
                    comp_dict[key] = value
                        
            info[f"comp_{i}"] = comp_dict
                
        info["p0"] = np.asarray(self.p0, dtype=float).tolist()
        info["filename_append"] = self.filename_append
        info["walkers"] = self.walkers
        info["steps"] = self.steps
        info["burntime"] = self.burntime
        #info["bounds"] = self.bounds
        
        best = dict()
        #all_units = np.asarray(["JY/PIX", "PIX", "PIX", "PIX", "PIX", "PIX", "PIX", "RAD", "NONE", "NONE"])
        param_keys = len(self.fits) * list(self.params['comp_0'].keys())
        param_keys = np.asarray(param_keys).reshape(len(self.fits), -1)
        for i in range(len(self.popt[self.prms])):
            best[param_keys[self.prms][i]] = {"value": self.popt[self.prms][i]}#, "unit": all_units[self.params.values][i]}
        
        info["initial"] = best
        info["data"] = self.sampler.tolist()
        
        with open(path.replace(".fits", ".json"), "w") as f:
            json.dump(info, f, indent=4)
    
    def load_json(self, path:str = ""):
        with open(path.replace(".fits", ".json"), "r") as f:
            info = json.load(f)
            
        self.walkers = info["walkers"]
        self.steps = info["steps"]
        self.burntime = info["burntime"]
        self.halo.name = info["object"]
        self.halo.path = info["filename"]
        self.filename_append = info["filename_append"]
        self.p0 = np.asarray(info["p0"])
        
        self.sampler = np.asarray(info["data"])
        self.samples = self.sampler[:, int(self.burntime):].reshape(
            (-1, self.dim)
        )
        
        for i, fit in enumerate(self.fits):
            fit.params = pd.DataFrame.from_dict(
                {"params": info[f"comp_{i}"]["params"].values()}, 
                orient = "index", 
                columns = info[f"comp_{i}"]["params"].keys()
            ).loc["params"]
            
            fit.frozen = pd.DataFrame.from_dict(
                {"frozen": info[f"comp_{i}"]["frozen"].values()}, 
                orient = "index", 
                columns = info[f"comp_{i}"]["frozen"].keys()
            ).loc["frozen"]
            
            fit.dim = info[f"comp_{i}"]["dim"]
            self.mask = info[f"comp_{i}"]["mask"]
        
        self.params = pd.concat([fit.params for fit in self.fits], axis=1)
        self.params.columns = [f"comp_{i}" for i in range(len(self.fits))]
        self.prms = self.params.values.T
        self.frozen = pd.concat([fit.frozen for fit in self.fits], axis=1)
        self.frozen.columns = [f"comp_{i}" for i in range(len(self.fits))]
        self.frzn = self.frozen.values.T
        
        popt = []
        param_keys = len(self.fits) * list(self.params['comp_0'].keys())
        param_keys = np.asarray(param_keys).reshape(len(self.fits), -1)
        for i in range(len(self.prms[self.prms])):
            if self.prms[self.prms][i]:
                popt.append(info["initial"][param_keys[self.prms][i]]["value"])
        
        self.popt = np.asarray(popt)
        
    
        
    def get_samples(self) -> np.ndarray:
        return self.samples.T
    
    def get_param_names(self) -> list:
        names = []
        for i, _ in enumerate(self.fits):
            params = self.params['comp_'+str(i)]
            frozen = self.frozen['comp_'+str(i)]
            names.append(list(params[params & ~frozen].keys()))
        return names
        
        
        
def Fit(_parent_: RadioHalo, model: list[str]|str = 'circle' ,**kwargs) -> MultiComponentFitting|SingleComponentFitting:
    if isinstance(model, str):
        return SingleComponentFitting(_parent_, model=model, **kwargs)
    elif isinstance(model, list):
        return MultiComponentFitting( _parent_, model=model, **kwargs)
    else:
        raise ValueError("model must be a string or a list of strings")
    
    
    
def pre_mcmc_func(obj, *theta):
    idx = 0
    compunent_sum = 0
    for i, fit in enumerate(obj.fits):
        model_theta = utils.add_labels(fit, theta[idx:idx+fit.dim])
        if obj.link[i]:
            if i == 0:
                link_x = model_theta['x0']
                link_y = model_theta['y0']
            if obj.link[i] and i > 0:
                model_theta['x0'] = link_x
                model_theta['y0'] = link_y
        model = fit._func_(obj, model_theta)
        compunent_sum += model[obj.image_mask.ravel()]
        idx += (i+1) * fit.dim 
    return compunent_sum


def load(path) -> SingleComponentFitting:
    with open(path, "r") as f:
        info = json.load(f)
    
    halo = RadioHalo(
        info["object"], 
        info["filename"], 
        mask_path=info["maskPath"], 
        #output_path=info["outputPath"],
        decreased_fov=info["cropped"],
        z=info["redshift"]
    )
    halo.logger.info("\n\nLoaded MCMC samples from file: %s\n" % path)
    
    frozen = pd.DataFrame.from_dict(
        {"frozen": info["frozen"].values()}, 
        orient = "index", 
        columns = info["frozen"].keys()
    ).loc["frozen"]
    
    frozen_vals = info["frozen_vals"]
    freeze = dict(zip(frozen[frozen].keys(), frozen_vals))
    
    fit = Fit(
        halo, 
        model=info["model_name"], 
        walkers=info["walkers"], 
        steps=info["steps"], 
        burntime=info["burntime"],
        freeze_params=freeze,
    )
    
    fit.params = pd.DataFrame.from_dict(
        {"params": info["params"].values()}, 
        orient = "index", 
        columns = info["params"].keys()
    ).loc["params"]
    
    fit.frozen = frozen
    fit.frozen_vals = np.asarray(frozen_vals)
    fit.p0 = np.asarray(info["p0"])
    
    fit.filename_append = info["filename_append"]
    
    fit.sampler = np.asarray(info["data"])
    fit.samples = fit.sampler[:, int(fit.burntime):].reshape(
        (-1, fit.dim)
    )
    
    popt = []
    for i in range(len(fit.params)):
        if fit.params[i]:
            popt.append(info["initial"][fit.params.keys()[i]]["value"])
    fit.popt = utils.add_labels(fit, np.asarray(popt))
    return fit
    
    