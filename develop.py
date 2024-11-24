import halo_fdca as fdca
import sys, os


Halo = fdca.RadioHalo(
    'A1033', 
    'Example/data_dir_test/A1033-HBA-sub-MFS-T50kpc.fits', 
    output_path='Example/Output_dir', 
    z=0.1220, 
    mask_path='Example/data_dir_test/A1033halo_hba.reg',
    decreased_fov=True
)



#fit = fdca.Fit(Halo, walkers=10, steps=100, model = ["circle", "rotated_ellipse"], link_loc=[True, True])
#fit.run()
#fit.save()
#fit.load()
#sys.exit()

# freeze parameters in image units
freeze = {
    "x0": 330.,
    "y0": 315.,
}

freeze = {"k_exp": 0.5}

fit = fdca.Fit(Halo, walkers=10, steps=100, model='circle', k_exponent=True)#, freeze_params=freeze)
#fit.run()
#fit.save()
fit.load()

chi2 = fit.results.get_chi2()
flux = fit.results.get_flux()
fit.results.plot()
