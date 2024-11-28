# Halo-FDCA (Flux Density CAlculator)

## Introduction
This software is created to automate flux density (and power) estimations of radio halos in galaxy clusters. This is done by fitting the surface brightness profile of halos to a mathematical model using Bayesian Inference. From the resulting fit, the flux density profile can be calculatied analytically. The text below provides a step-by-step example and walk trough of the algorithm as well as a summary of installation and machine requirements. A full text on the methodology can be found under 'Citation'.

This software is open source and still in development. Suggestions, remarks, bugfixes, questions etc. are more than welcome and can be sent to j.boxelaar@ira.inaf.it.

### Citation
If you make use of this code in its original form or portions of it, please cite:<br>
_Boxelaar J.M. et al., A robust model for flux density calculations of radio halos in galaxy clusters: Halo-FDCA, Astronomy and Computing, Volume 35, 2021, 100464, ISSN 2213-1337, https://doi.org/10.1016/j.ascom.2021.100464._

Also available on the arXiv: https://arxiv.org/abs/2103.08554

## Requirements

This software is written and tested in the `Python 3.11`(!!) and is meant for application in astronomical research. **Note:** Multiprocessing does not work for python versions below 3.11, in case you use an older vresion of Python, always set `num_cpus=1` in the fdca.Fit() function. 

Key software requirements to run the pipeline include:

`Astropy` v4.0 and up<br>
https://www.astropy.org

`Scipy` v1.4 and up<br>
`Numpy` v1.19 and up<br>
`Matplotlib` v3.2 and up<br>
https://www.scipy.org/docs.html

`emcee` v3.0 and up<br>
https://emcee.readthedocs.io/en/stable/ 

`corner` v2.2 <br>
https://corner.readthedocs.io/en/latest/index.html 

`skimage` v0.17 <br>
https://scikit-image.org

`tqdm`<br>
https://tqdm.github.io/ (this is not a strict requirement, but it is useful if you want to follow the progress of the run)

This software will run a multithreaded process by default using the `multiprocessing` module. This requires significant computing power and it is advised to use a dedicated external computing mashine. Multithreading can be turned off but this will increase the run time up to a factor 10 (depending on the settings). 

The code is tested on Unix machines using both PYTHON 3.6 and PYTHON 3.8. Please inform us when you run into any issues related toincompatible software versions. 

## Algorithm Instructions
### Overview
![Flowchart!](flowchart-1.png "Flowchart")

This flowchart gives a general overview of the pipeline. On code level, the pipeline works roughly like this:<br>
First, a `Radio_Halo` object must be initiated. When initiating the object, all relevant properties are processed. This class is documented in `HaloObject.py`.
This class also handles the very first fit such that the image coordinates can be related to sky coordinates without using the header. The `Radio_Halo` class handles steps up to 'Second preliminary fit'. 'Rotating and regridding' is not performed by the class.

The Markov Chain Monte Carlo (MCMC) algorithm is performed in the blue part of the flowchart. This takes place in the `fitting` class in `markov_chain_monte_carlo.py` (see class documentation there and below). This class takes a Radio Halo object as input and from there starts the profile fitting based on the extra settings given as input. 'Chains' that are the result of MCMC are saved in new FITS files (**./outputpath/Samples/**), settings of the speciffic run are saved in the header. 

The `processing` class in `mcmc_fitting_multicomponent.py` takes a halo object as input and processes the MCMC results by generating figures, statistical analysis and final flux density and parameter estimations. these results are found in the log files.

### Input
The code requires very specific input to be able to work properly. 

- .FITS file:<br>
This includes a FITS image with a clean, preferably point source subtracted radio image of a galaxy cluster containing diffuse emission. 

- .reg file:<br>
A DS9 region file with regions drawn around contaminating sources/artifacts in the image, saved in physical coordinates. The shape of the region can be of any kind. A region file is not mandatory to run the code. 

Currently, cluster characteristics such as location and redshift can be retrieved from catalogues on VIZIER. This automatic retrieval of cluster information is availible for MCXC, PSZ2, Abell and WHL clusters. A catalogue search is not always succesful, it is adviced to give the essential cluster information, redshift and sky location, as input. 

### Getting Started
In v2.0, Halo-FDCA is used as a package. See `develop.py` as example of how to run Halo-FDCA. we are currently working on making the package availible through pip. For now, if one wants to use the code as a package, run `export PYTHONPATH=/path_to_code/halo_fdca:$PYTHONPATH` in your terminal. 

Creating a RadioHalo object passing the minimum amount of information:

```python
import halo_fdca as fdca

Halo = fdca.RadioHalo('A2744','Example/Data_dir/A2744_JVLA.image.fits')
```

Creating a Fitting object and start a run:
```python
import halo_fdca as fdca

Halo = fdca.RadioHalo('A2744','Example/Data_dir/A2744_JVLA.image.fits')

# in Fit(), it is possible to specify number of steps and walkers 
fit = fdca.Fit(Halo, model='circle')
fit.run()
```

To prevent rerunning the MCMC, one can save a run with `fit.save(path)` where the path to the saved samples is optional. Load a previous run with 

```python
fit = fdca.Fit(Halo, model='circle').load()
# OR
fit = fdca.load('samples.json')
```
where the path points to the json generated by fit.save().

Analyzing results after a fit is run or loaded is done by

```python

fit = fdca.Fit(Halo, model='circle').load() 

fit.results.plot()
chi2 = fit.results.get_chi2()
flux, flux_uncertainty = fit.results.get_flux()
power, power_uncertainty = fit.results.get_power()

samples = fit.get_samples()
parameter_names = fit.get_param_names()
```


### Output
Samples file in json format containing the walker strings and run information (found in **outputpath/Samples**). This file is used to process the routine and retrieve flux denisty and parameter values. All values and their one sigma uncertainties will be printed in the log file after running the pipeline. The figures will be in **outputpath/Plots**. There, all relevant figures will be saved. This includes the corner plot, walker plot and radio image with model overlay for the original and regridded image. 

The .log file outputs the final fit information and it looks something like this:
```
Run information for object A1033:
    RMS noise: 0.0002863803820218891 Jy / beam
    Model: circle
    Walkers: 10
    Steps: 100
    Burntime: 12
    Mask: True
    Rebin: True
    K_exponent: True
    Offset: False

Fit results:
    Flux density at 143.7 MHz: 336.81546 mJy +/- 4.79004 mJy
    Reduced chi-squared: 4.859490839643051
    I0:   25.543 +/- 0.2336 ($\mu$Jy arcsec$^{-2}$)
    x0:   157.928 +/- 0.0002 (deg)
    y0:   35.040 +/- 0.0002 (deg)
    r1:   100.547 +/- 1.1243 (kpc)
    
    Uncertainties (lower, upper):
        [2.38880646e-01 1.90126966e-04 1.85981806e-04 1.27793972e+00
 5.43166949e-20]
        [2.28331040e-01 2.48196633e-04 2.07639059e-04 9.70618808e-01
 8.45588410e-20]
```

The same information can be printed to the terminal by printing the resuts: `print(fit.results)`. 

## Multi-component Fitting (Beta)

Fitting multiple exponential shapes to an image in now possible, but untested. The ability to add multiple components adds to possibility to link the location of the shapes (usefull for mega halos) or to change the profile of the exponential (e.g. to a Gaussian istead of a exponential., usefull for fitting the contribution from an AGN). As a example, given a RadioHalo, we want to fit a circle model with exponential shape and have a circle model with a Gaussian shape at the same location accounting for the mega halo shape:

```python
fit = fdca.Fit(Halo, model=['circle','circle'], link_loc=[True, True], profiles=["default", "gaussian"])
```

where `"default"` is the default exponential shape. **Note:** it is not yet possible to get results from the fits, to analyse the results, use the outputted samples directly:

```python
samples = fit.get_samples()
param_names = fit.get_param_names()
```
