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


fit = fdca.Fit(Halo, model=["circle", "rotated_ellipse"], walkers=50, steps=500)
sys.exit()







fit = fdca.Fit(Halo, walkers=10, steps=100)
fit.run()
fit.save()
fit.results.plot()
print(fit.results)


sys.exit()
#fit.run(save=True)

results = fdca.Processing(fit)

results.plot()
chi2 = results.get_chi2_value()
flux, flux_err = results.get_flux() # error is one sigma (68%).
power, power_err = results.get_power()
print(results)



sys.exit()


multiFit = fdca.MultiWavelengthFitting([Halo_low, Halo_high])
#multiFit.run(save=True)

results = fdca.MultiWavaelenghtProcessing(multiFit)

for processing in results.results:
    processing.plot()
    processing.get_chi2_value()
    processing.get_flux()# error is one sigma (68%).
    processing.get_power()

    print(processing)
