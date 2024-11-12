import halo_fdca as fdca
import sys, os


Halo = fdca.RadioHalo(
    'A1033', 
    'Example/data_dir_test/A1033-HBA-sub-MFS-T50kpc.fits', 
    output_path='Example/Output_dir', 
    z=0.122, 
    mask_path='Example/data_dir_test/A1033halo_hba.reg',
    decreased_fov=True
)

# freeze parameters in image units
freeze = {
    "x1": 330.,
    "y0": 315.,
}

fit = fdca.Fit(Halo, walkers=10, steps=100, freeze_params=freeze)
#fit.run()
#fit.save()
fit.load()

chi2 = fit.results.get_chi2()
flux = fit.results.get_flux()
fit.results.plot()
