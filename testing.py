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
    #decreased_fov=True
)

fit = fdca.Fitting(Halo_high, model="skewed")
fit.run()

multiFit = fdca.MultiWavelengthFitting([Halo_low, Halo_high])
multiFit.run(save=True)

#fdca.Fitting(Halo_low).run(save=True)
