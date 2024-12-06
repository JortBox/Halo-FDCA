#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: J.M. Boxelaar
Version: 08 June 2020
"""

from __future__ import division
from logging import Logger
import sys

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
#from .mcmc_fitting import Fitting
#from .mcmc_fitting_multicomponent import SingleComponentFitting, MultiComponentFitting


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
        fit,
        logger: Logger=None,
        save=False,
    ):
        self.data = fit.halo.data
        self.rebin = fit.rebin
        self.filename_append = fit.filename_append
        
        if hasattr(fit, 'sampler'):
            self.sampler = fit.sampler
            #self.info = fit.info
        else:
            logger.error("No sampler found in fit object. Exiting.")
            sys.exit()

        if logger is None:
            self.logger: Logger = fit.logger
        else:
            self.logger = logger
            
        self.save = save
        
        x = np.arange(0, self.data.shape[1], 1)
        y = np.arange(0, self.data.shape[0], 1)
        self.x_pix, self.y_pix = np.meshgrid(x, y)

        self.check_settings(fit)
        self.retreive_mcmc_params(fit)
        self.set_labels_and_units()

        self.dof = len(self.data.value.flat) - self.dim
        self.__repr__()
        
        
    def __repr__(self) -> str:
        
        uncertainties1 = self.percentiles_units[:, 1] - self.percentiles_units[:, 0]
        uncertainties2 = self.percentiles_units[:, 2] - self.percentiles_units[:, 1]
        errors = np.mean([uncertainties1, uncertainties2], axis=0)
        flux, flux_err = self.get_flux(debug=True)
        
        param_string = ""
        i = 0
        for param in range(len(self.params)):
            if self.frozen[param]:
                frozen = "   FROZEN"
                error = ''
                unit = f"({self.units[i]})" 
            elif not self.params[param]:
                frozen = ""
                error = ""
                unit = ""
            else:
                frozen = ""
                error = f" +/- {errors[i]:.4f}"
                unit = f"({self.units[i]})" 
                i += 1
                
            if self.params.values[param]: 
                param_string += f"{np.asarray(self.paramNames)[param]}:   {self.params_units[param]:.3f}{error} {unit}{frozen}\n    "
        
        run_details = f"""Run information for object {self.halo.name}:
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
    Reduced chi-squared: {self.get_chi2(debug=True)}
    {param_string}
    Uncertainties (lower, upper):
        {uncertainties1}
        {uncertainties2}
    """
        self.logger.debug(run_details)
        return run_details

    def plot(self):
        plot_fits.fit_result(
            self,
            self.model,
            self.halo.data,
            self.halo.rmsnoise * (self.halo.beam / self.halo.beam2pix),
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
        plot_fits.samplerplot(self)
        plot_fits.cornerplot(self)

    def check_settings(self, fit):
        self.model_name = fit.model_name
        self.paramNames = fit.paramNames
        self._func_ = fit._func_
        self.AppliedParameters = fit.AppliedParameters
        self.mask = fit.mask

        if fit.k_exponent:
            self.AppliedParameters[-2] = True
        if fit.offset:
            self.AppliedParameters[-1] = True

        self.params = pd.DataFrame.from_dict(
            {"params": self.AppliedParameters}, orient="index", columns=self.paramNames
        ).loc["params"]
        #self.dim = len(self.params[self.params])
        self.image_mask = utils.masking(fit.halo)

    def at(self, parameter):
        par = np.array(self.paramNames)[self.params & ~self.frozen]
        return np.where(par == parameter)[0][0]

    def retreive_mcmc_params(self, fit):
        self.noise = fit.noise
        self.rms = fit.rms
        self.halo = fit.halo
        self.alpha = fit.halo.alpha  # spectral index guess
        self.mask_treshold = fit.mask_treshold
        self.dim = fit.dim
        self.frozen = fit.frozen
        self.walkers = fit.walkers
        self.steps = fit.steps
        self.popt = fit.popt
        self.burntime = fit.burntime
        self.samples = fit.samples
        
        self.fit = fit
        
        percentiles = self.get_percentiles(self.samples)
        self.parameters = utils.add_labels(self.fit, percentiles[:, 1].reshape(self.dim))
        
        if "x0" in fit.frozen.keys():
            self.parameters["x0"] -= self.halo.margin[2]
        if "y0" in fit.frozen.keys():
            self.parameters["y0"] -= self.halo.margin[0]
        
        self.centre_pix = np.array(
            [self.parameters["x0"], self.parameters["y0"]], dtype=np.int64
        )

        self.model = self._func_(self, self.parameters).reshape(self.x_pix.shape) * u.Jy


    def get_percentiles(self, samples):
        percentiles = np.ones((samples.shape[1], 3))
        for i in range(samples.shape[1]):
            percentiles[i, :] = np.percentile(samples[:, i], [16, 50, 84])

        if self.model_name in ["rotated_ellipse", "skewed"] and "ang" not in self.params[self.frozen].keys():
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
                self.logger.error("Angle matching failed in processing.get_percentiles. continuing with default.")
                ang = np.percentile(samples[:, self.at("ang")], [16, 50, 84])

            percentiles[self.at("ang"), :] = ang
        return percentiles

 

    def set_labels_and_units(self):
        self.samples_units = self.samples.copy()
        samples_units = self.samples.copy()
        samples_list = np.asarray([samples_units[:, i] for i in range(self.dim)])
        transformed = utils.transform_units(
            self, 
            np.copy(samples_list), 
            unlabeled=True, 
            keys=self.params[self.params & ~self.frozen].keys()
        )
        for i in range(self.dim):
            self.samples_units[:, i] = transformed[i]

        
        self.params_units = utils.transform_units(self, self.parameters)
        self.percentiles_units = self.get_percentiles(self.samples_units)


        self.get_units()
        uncertainties1 = self.percentiles_units[:, 1] - self.percentiles_units[:, 0]
        uncertainties2 = self.percentiles_units[:, 2] - self.percentiles_units[:, 1]
                     
        string_to_print = f"\n Parameters: \n{str(self.params_units[self.params])} \
            \nOne sigma parameter uncertainties (lower, upper): \
            \n{str(uncertainties1)} \n{str(uncertainties2)} \
            \nIn Units: {str(self.units)}" 
                                    
        self.logger.debug(string_to_print)
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

        self.labels_units = np.copy(self.labels)#[:self.dim]
        #frozen = self.frozen[self.params]
        for i in range(len(self.labels)):
            #self.labels_units[i] = self.labels[~frozen][i] + " [" + self.units[~frozen][i] + "]"
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
                self.logger.debug(
                    "{}% Confidence interval of {}: ({:.5f}, {:.5f}) {} \n".format(
                        percentage,
                        self.labels[i],
                        conf_low[i],
                        conf_up[i],
                        self.units[i],
                    )
                )
        else:
            for i in range(self.dim):
                se[i] = np.sqrt(
                    np.mean(self.samples[:, i] ** 2.0)
                    - np.mean(self.samples[:, i]) ** 2.0
                )
            conf_low = self.parameters - z_alpha * se
            conf_up = self.parameters + z_alpha * se
            for i in range(self.dim):
                self.logger.debug(
                    "{}% Confidence interval of {}: ({:.5f}, {:.5f}) \n".format(
                        percentage, self.labels[i], conf_low[i], conf_up[i]
                    )
                )
        return [conf_low, conf_up]
    
    def set_data_to_use(self, data) -> np.ndarray:
        if self.rebin:
            binned_data = utils.regridding(self.halo, data, decrease_fov=True)
                
            self.binned_image_mask = utils.regridding(
                self.halo, 
                self.image_mask.astype(int) * u.Jy, 
                mask=not self.halo.cropped
            ).value
            use = binned_data.value
            return use.ravel()[
                self.binned_image_mask.ravel()
                >= self.mask_treshold * self.binned_image_mask.max()
            ]
        else:
            if self.mask:
                return self.data.value.ravel()[self.image_mask.astype(int).ravel() <= 0.5]
            else:
                return self.data.value.ravel()

    def get_chi2(self, mask_treshold: float = None, debug=False) -> float:
        """_summary_

        Args:
            mask_treshold (float, optional): _description_. Defaults to None.
            debug (bool, optional): _description_. Defaults to False.

        Returns:
            float: _description_
        """
        mask_treshold = self.mask_treshold if mask_treshold is None else mask_treshold
        
        x = np.arange(0, self.halo.data_mcmc.shape[1], 1)
        y = np.arange(0, self.halo.data_mcmc.shape[0], 1)
        self.x_pix, self.y_pix = np.meshgrid(x, y)

        params = self.parameters.copy()
        #print(params)
        #sys.exit()
        params[1] += self.halo.margin[2]
        params[2] += self.halo.margin[0]

        binned_data = self.set_data_to_use(self.halo.data_mcmc)
        model = (self._func_(self, params, rotate=True).reshape(self.halo.data.shape) * u.Jy)
        binned_model = utils.regrid_to_beamsize(self.halo, model)
        rmsregrid = utils.findrms(binned_data)

        binned_image_mask = utils.regridding(
            self.halo, 
            self.image_mask.astype(int) * u.Jy, 
            mask=not self.halo.cropped
        ).value
        binned_model = binned_model.ravel()[
            binned_image_mask.ravel() >= mask_treshold * binned_image_mask.max()
        ]
       
        
        chi2 = np.sum(((binned_data - binned_model) / rmsregrid) ** 2.0)
        binned_dof = len(binned_data) - self.dim
        self.chi2_red = chi2 / binned_dof
        
        
        ln_likelihood = - 0.5 * np.sum( ((binned_data - binned_model) / rmsregrid) ** 2 + np.log(2 * np.pi * (rmsregrid) ** 2)) 
        AIC = 2 * (self.dim - ln_likelihood)

        if not debug:
            self.logger.debug(f"chi^2: {chi2:.2f}")
            self.logger.debug(f"effective DoF: {binned_dof:d}")
            self.logger.debug(f"chi^2_red: {self.chi2_red:.2f}")
            self.logger.debug(f"AIC: {AIC:.2f}")

        x = np.arange(0, self.data.shape[1], 1)
        y = np.arange(0, self.data.shape[0], 1)
        self.x_pix, self.y_pix = np.meshgrid(x, y)
        return self.chi2_red

    def get_flux(self, int_max: float = np.inf, freq:float=None, alpha:float=None, debug=False) -> tuple[float, float]:
        """
        Calculate flux density of the halo at a given frequency. Based on the MCMC samples.

        Args
        -------
        int_max : float
            max integration radius in units of e-folding. Defaults to np.inf.
            
        freq : float 
            frequency in MHz. Defaults to None.
            
        alpha : float 
            spectral index. Required when calculating flux at different frequency. Defaults to None.
            
        debug : bool
            If True, do not print flux to terminal. Defaults to False.

        Returns
        -------
        flux : tuple[float, float] 
            Flux density and its uncertainty in mJy.
        """
        alpha = self.alpha if alpha is None else alpha
        freq = self.halo.freq if freq is None else freq*u.MHz
        
        samples = dict(zip(self.paramNames, self.parameters))
        for i in range(self.dim):
            samples.update({self.params[self.params & ~self.frozen].keys()[i]: self.samples[:, i]})
            
        for i, key in enumerate(self.frozen[self.frozen].keys()):
            samples.update({key: self.fit.frozen_vals[i]})

        a = samples['r1'] * self.halo.pix_size
        if self.model_name == "skewed":
            b = samples['r2'] * self.halo.pix_size
            c = samples['r3'] * self.halo.pix_size
            d = samples['r4'] * self.halo.pix_size
            factor = a * b + c * d + a * d + b * c
        elif self.model_name in ["ellipse", "rotated_ellipse"]:
            b = samples['r2'] * self.halo.pix_size
            factor = 4 * a * b
        else:
            factor = 4 * a**2
    
        m = samples['k_exp'] + 0.5
        I0 = u.Jy * samples['I0'] / self.halo.pix_area
        flux = (
            gamma(1.0 / m)
            * np.pi
            * I0
            / (4 * m)
            * factor
            * gammainc(1.0 / m, int_max ** (2 * m))
            * (freq / self.halo.freq) ** self.alpha
        ).to(u.mJy)
        percentiles = np.percentile(flux, [16, 50, 84])
        flux_val = percentiles[1]
        flux_err = ((percentiles[2] - percentiles[1]) + (percentiles[1] - percentiles[0])) / 2.0
        
        self.flux = np.copy(flux)
        self.flux_freq = freq
        
        signal_to_noise = I0.mean().to(u.Jy/self.halo.beam) / self.halo.rmsnoise
        if signal_to_noise < 10:
            self.logger.warning(f"Halo peak below 10 x RMS, Fitting probably unreliable. Value: {signal_to_noise:.2f}")

        if not debug:
            self.logger.info(f"MCMC Flux at {freq.value:.1f} {freq.unit}: {flux_val.value:.2f} +/- {flux_err.value:.2f} {flux.unit}")
            self.logger.debug(f"Flux integration radius {int_max}")
            self.logger.debug(f"S/N based on flux (Flux / Flux Error) {flux_val.value / flux_err.value:.2f}")
            self.logger.debug(f"Signal to noise (I_0 / RMS): {signal_to_noise}")
        return flux_val, flux_err

    def get_power(self, freq:float=None, alph:float=None) -> tuple[float, float]: 
        """
        Calculate radio power of the halo at a given frequency. Based on the MCMC samples.

        Args
        -------
        freq : float 
            frequency in MHz. Defaults to None and takes from halo fits file.
            
        alpha : float 
            spectral index. REquired when calculating flux at different frequency. Defaults to None and takes from halo fits file.
            
        debug : bool
            If True, do not print flux to terminal. Defaults to False.

        Returns
        -------
        flux : tuple[float, float] 
            Flux density and its uncertainty in mJy.
        """
        alpha = self.alpha if alpha is None else alpha
        freq = self.halo.freq if freq is None else freq

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

        percentiles = np.percentile(power, [16, 50, 84])
        power_val = percentiles[1]
        power_err = ((percentiles[2] - percentiles[1]) + (percentiles[1] - percentiles[0])) / 2.0
        
        self.logger.info(
            "Power at {:.1f} {}: ({:.3g} +/- {:.3g}) {}".format(
                freq.value,
                freq.unit,
                power_val.value,
                power_err.value,
                power.unit,
            ),
        )
        return power_val, power_err

    def get_radius_estimate(self) -> float|None:
        if self.model_name == "circle":
            radius = - self.parameters["r1"] *np.log(3 * self.rms.value / self.parameters["I0"]) * self.halo.pix2kpc
            return radius
        else:
            return None