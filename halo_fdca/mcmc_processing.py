#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: J.M. Boxelaar
Version: 08 June 2020
"""

from __future__ import division
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats

from scipy.special import gammainc, gamma
from astropy import wcs
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits


# Subfile imports
from . import plot_fits
from . import fdca_utils as utils
from .mcmc_fitting import Fitting


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
        fit: Fitting,
        rebin=True,
        logger=None,
        save=False,
        sampler=None,
        sample_info=None,
    ):
        
        self.data = fit.halo.data

        if logger is None:
            self.log = fit.log
        else:
            self.log = logger
        
        x = np.arange(0, self.data.shape[1], 1)
        y = np.arange(0, self.data.shape[0], 1)
        self.x_pix, self.y_pix = np.meshgrid(x, y)

        self.log.log(logging.INFO, "Model name: {}".format(fit.model_name))
        self.noise = fit.noise
        self.rms = fit.rms
        self.save = save
        self.halo = fit.halo
        self.fit = fit
        self.alpha = fit.halo.alpha  # spectral index guess
        self.mask_treshold = 0.5

        self.check_settings(fit.model_name, fit.mask)
        self.extract_chain_file(rebin, sampler, sample_info)
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
    K_exponent: {self.fit.k_exponent}
    Offset: {self.fit.offset}

Fit results:
    Flux density at {self.halo.freq:.1f}: {flux:.5f} +/- {flux_err:.5f}
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
            self.halo.data_mcmc,
            self.halo.rmsnoise,
            mask=self.mask,
            regrid=True,
        )
        plot_fits.samplerplot(self)
        plot_fits.cornerplot(self)

    def check_settings(self, model, mask):
        self.model_name = model
        self.paramNames = self.fit.paramNames
        self._func_ = self.fit._func_
        self.AppliedParameters = self.fit.AppliedParameters

        if self.fit.k_exponent:
            self.AppliedParameters[-2] = True
        if self.fit.offset:
            self.AppliedParameters[-1] = True

        self.params = pd.DataFrame.from_dict(
            {"params": self.AppliedParameters}, orient="index", columns=self.paramNames
        ).loc["params"]
        self.dim = len(self.params[self.params])
        self.image_mask, self.mask = utils.masking(self, mask)


    def extract_chain_file(self, rebin, sampler, info):
        filename_append = "_{}".format(self.model_name)
        if self.mask:
            filename_append += "_mask"
        # if rebin: filename_append += '_rebin'
        if self.fit.k_exponent:
            filename_append += "_exp"
        if self.fit.offset:
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

 

    def set_labels_and_units(self):
        self.samples_units = self.samples.copy()
        samples_units = self.samples.copy()
        samples_list = list()

        #x0 = np.percentile(self.samples.real[:, 1], [16, 50, 84])[1] + self.halo.fov_info_mcmc[2]
        #y0 = np.percentile(self.samples.real[:, 2], [16, 50, 84])[1] + self.halo.fov_info_mcmc[0]
    
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
        if self.fit.k_exponent:
            labels.append("$k$")
            units.append(" ")
            fmt.append(".3f")
        if self.fit.offset:
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
            
        if self.fit.k_exponent:
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

        self.log.log(
            logging.INFO,
            f"MCMC Flux at {freq.value:.1f} {freq.unit}: \
                {self.flux_val.value:.2f} +/- {self.flux_err.value:.2f} \
                {flux.unit}"
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

    def get_radius_estimate(self):
        if self.model_name == "circle":
            radius = - self.parameters["r1"] *np.log(3 * self.rms.value / self.parameters["I0"]) * self.halo.pix2kpc
            print(radius)
        else:
            pass