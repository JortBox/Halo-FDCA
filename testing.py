import halo_fdca as fdca

Halo_low = fdca.RadioHalo(
    'A1033', 
    'Example/data_dir_test/A1033-LBA-sub-MFS-image.fits', 
    output_path='Example/Output_dir', 
    z=0.1220, 
    mask_path='Example/data_dir_test/A1033halo_lba.reg'
)

Halo_high = fdca.RadioHalo(
    'A1033', 
    'Example/data_dir_test/A1033-HBA-sub-MFS-T50kpc.fits', 
    output_path='Example/Output_dir', 
    z=0.1220, 
    mask_path='Example/data_dir_test/A1033halo_hba.reg',
    decreased_fov=True
)



multiFit = fdca.MultiWavelengthFitting([Halo_low, Halo_high])
#multiFit.run(save=True)

results = fdca.MultiWavaelenghtProcessing(multiFit)

for processing in results.results:
    processing.plot()
    processing.get_chi2_value()
    processing.get_flux()# error is one sigma (68%).
    processing.get_power()

    print(processing)
