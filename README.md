# Halo-FDCA (Flux Density CAlculator)

## Introduction
This software pipile is created to automate flux density (and power) estimations of radio halos in galaxy clusters. This is done by fitting the surface brightness profile of halos to a mathematical model using Bayesian Inference. From the resulting fit, the flux density profile can be calculatied analytically. The text below provides a step-by-step example and walk trough of the algorithm as well as a summary of installation and machine requirements. A full text on the methodology can be found under 'Citation'.

This software is open source and still in development. Suggestions, remarks, bugfixes, questions etc. are more than welcome and can be sent to boxelaar@strw.leidenuniv.nl .


### Citation
If you make use of this code in its original form or portions of it, please cite:<br>
Boxelaar et al., 2020 (in prep.), Astronomy and Computing

Also available on the arXiv: t.b.a.

## Requirements

This software is written in the `Python` programming language and is meant for application in astronomical research. Key software requirements to run the pipeline include:

`Astropy` v4.0 and up<br>
https://www.astropy.org

`Scipy` v1.4 and up<br>
`Numpy` v1.19 and up<br>
`Matplotlib` v3.2 and up<br>
https://www.scipy.org/docs.html

`emcee` v3.0 and up<br>
https://emcee.readthedocs.io/en/stable/ 

`corner`<br>
https://corner.readthedocs.io/en/latest/index.html 

`skimage` v0.17 <br>
https://scikit-image.org


This software will run a multithreaded process by default using the `multiprocessing` module. This requires significant computing power and it is advised to use a dedicated external computing mashine. Multithreading can be turned off but this will increase the run time up to a factor 10 (depending on the settings).  

## Algorithm Instructions
### Overview
![Flowchart!](flowchart-1.png "Flowchart")

This flowchart gives a general overview of the pipeline. On code level, the pipeline works roughly like this:<br>
First, a `Radio_Halo` object must be initiated. When initiating the object, all relevant properties are processed. This class is documented in `HaloObject.py`.
This class also handles the very first fit such that the image coordinates can be related to sky coordinates without using the header. The `Radio_Halo` class handles steps up to 'Second preliminary fit'. 'Rotating and regridding' is not performed by the class.

The Markov Chain Monte Carlo (MCMC) algorithm is performed in the blue part of the flowchart. This takes place in the `fitting` class in `markov_chain_monte_carlo.py` (see class documentation there and below). This class takes a Radio Halo object as input and from there starts the profile fitting based on the extra settings given as input. 'Chains' that are the result of MCMC are saved in new FITS files (`./Results/Samples/`), settings of the speciffic run are saved in the header. 

The `processing` class in `markov_chain_monte_carlo.py` takes a halo object as input and processes the MCMC results by generating figures, statistical analysis and final flux density and parameter estimations. these results are found in the log files: `./outputpath/log/`.

### Input
The code requires very specific input to be able to work properly. 

- .FITS file:<br>
This includes a FITS image with a clean, preferably point source subtracted radio image of a galaxy cluster containing diffuse emission. 

- .reg file:<br>
A DS9 region file with regions drawn around contaminating sources/artifacts in the image, saved in physical coordinates. The shape of the region can be of any kind. A region file is not mandatory to run the code. 

Currently, cluster characteristics such as location and redshift can be retrieved from catalogues on VIZIER. This automatic retrieval of cluster information is availible for MCXC, PSZ2, Abell and WHL clusters. A catalogue search is not always succesful, it is adviced to give the essential cluster information, redshift and sky location, as input. 

### Settings
Halo-FDCA is very flexible and has many settings that can be easily set by the user within the terminal using `argparse`. All possible settings and their use can be inspected by entering `python3 HaloFitting.py -h` in the terminal. The help page will look like this:

```
usage: HaloFitting.py [-h] [-model {circle,ellipse,rotated_ellipse,skewed}]
                      [-frame FRAME] [-loc LOC] [-z Z] [-m M] [-m_file M_FILE]
                      [-out_path OUT_PATH] [-fov FOV] [-spectr_idx SPECTR_IDX]
                      [-walkers WALKERS] [-steps STEPS] [-burntime BURNTIME]
                      [-max_radius MAX_RADIUS] [-gamma_prior GAMMA_PRIOR]
                      [-k_exp K_EXP] [-s S] [-run_mcmc RUN_MCMC]
                      [-int_max INT_MAX] [-freq FREQ]
                      object d_file

Halo-FDCA: An automated flux density calculator for radio halos in galaxy
clusters. (Boxelaar et al.)

positional arguments:
  object                (str) Cluster object name
  d_file                (str) FITS image location (containing radio halo).

optional arguments:

  -h, --help            show this help message and exit
  -model {circle,ellipse,rotated_ellipse,skewed}
                        (str) Model to use. choose from (circle, ellipse,
                        rotated_ellipse, skewed). Default: circle
  -frame FRAME          (str) Coordinate frame. Default: ICRS
  -loc LOC              (str) Sky coordinates of cluster. provide coordinates
                        of the form: 'hh mm ss.ss -dd mm ss.s' in hourangle
                        units. Default: None and image centre is chosen.
  -z Z                  (float) cluster redshift
  -m M                  (bool) choose to include mask or not. If True,
                        -maskPath should be specified. Default: True
  -m_file M_FILE        (str) Mask file location. Default: None
  -out_path OUT_PATH    (str) Path to code output. Default: directory code is
                        in.
  -fov FOV              (bool) Declare if image size has to be decreased
                        before MCMC-ing. Amount of decreasement has ben
                        automatically set to 3.5*r_e. Default: True
  -spectr_idx SPECTR_IDX
                        (float) Set spectral index of cluster (S ~ nu^alpha).
                        Used to calculate power and extrapolate flux to
                        arbitrary frequencies. Default: -1.2
  -walkers WALKERS      (int) Number of walkers to deploy in the MCMC
                        algorithm. Default: 200
  -steps STEPS          (int) Number of evauations each walker has to do.
                        Default: 1200
  -burntime BURNTIME    (int) Burn-in time for MCMC walkers. See emcee
                        documentation for info. Default: None. this is 1/4th
                        of the steps.
  -max_radius MAX_RADIUS
                        (float) Maximum posiible radius cut-off. Fitted halos
                        cannot have any r > max_radius. In units of kpc.
                        Default: None (implying image_size/2).
  -gamma_prior GAMMA_PRIOR
                        (bool) Wether to use a gamma distribution as a prior
                        for radii. Default is False. For the gamma parameters:
                        shape = 2.5, scale = 120 kpc. Default: False
  -k_exp K_EXP          (bool) Wether to use k exponent to change shape of
                        exponential distribution. Default: False
  -s S                  (bool) Whether to save the mcmc sampler chain in a
                        fits file. Default: True.
  -run_mcmc RUN_MCMC    (bool) Whether to run a MCMC routine or skip it to go
                        straight to processing. can be done if a runned sample
                        already exists in the output path. Default: True
  -int_max INT_MAX      (float) Integration radius in r_e units. Default: inf
  -freq FREQ            (float) frequency in MHz to calculate flux in. When
                        given, the spectral index will be used. Default: image
                        frequency        
```
The keywords `object` and `d_path` are mandatory to give to be able to run the code.

Note: At this point, the code only works when `HaloFitting.py` is run from its current directory. To get around this problem, the FDCA directory should be placed in your python "site-packages" directory to effectively let it function as a package. 

### Output
Samples file in FITS format containing the walker strings and run information (found in **outputpath/Output/Samples**). This file is used to process the routine and retrieve flux denisty and parameter values. All values and their one sigma uncertainties will be printed in the log file after running the pipeline. The figures will be in **outputpath/Output/Plots**. There, all relevant figures will be saved. This includes the corner plot, walker plot and radio image with model overlay for the original and regridded image. 

## Example (Abell 2744)

### Installation and Example (Abell 2744)

We will now show an example to showcase the code using Abell 2744. See https://ui.adsabs.harvard.edu/abs/2017ApJ...845...81P  by C.J.J Pearce, (2017) for more information on the cluster.

The recommended way to install prefactor is to download it from github with:

`git clone https://github.com/JortBox/Halo-FDCA.git`
This allows for easy updating of the code to include bugfixes or new features. Once downloaded, the installation is complete; to set up a run from the **Example** directory, use the following line included with standard settings:
```
python3 HaloFitting.py Abell2744 ./Example/A2744_JVLA.image.fits -m_file ./Example/A2744halo.reg -loc '00 14 20.03 -30 23 17.8' -z 0.308 -out_path ./Example/ -model circle
```
Here, the circular model is fitted and samples/figures are saved to the Example directory. The relevant data is located in **./Example/A2744_JVLA.image.fits** and the mask is in **./Example/Masks/A2744halo.reg**.

