import numpy as np

from photon_generator.photon_generator import FlatFieldGenerator, MoffatPSFGenerator, \
    GaussianPSFGenerator, SeeingPSFGenerator, DiffractionPSFGenerator, \
    SourcePhotonGenerator

test_star_field_size_x = 32
test_star_field_size_y = 32
test_frame_count =10
pdf_sum_tolerance=0.001
photon_count_tolerance = 0.05


# Flat
test_flat_photon_count = 1

# Moffat
test_alpha = 4
test_beta = 1.5

# Gaussian
test_sigma_x = 1
test_sigma_y = 1
test_total_photon = 2000

# Poisson Sampling
test_poisson_mean_tolerance = 0.2

# Seeing and Diffraction
test_n_screen_size = 256 # This is size of phase screen (NxN)
test_telescope_diameter = 8. # Diameter of telescope
test_pixel_scale = test_telescope_diameter/test_n_screen_size # Size of each phase pixel in metres
test_r0 = 0.20 # Fried parameter (metres) 
test_L0 = 100 # Outer scale (metres)
test_stencil_length_factor = 32 # How much longer is the stencil that the desired phase
test_wavelength_nm = 500
test_padding_factor = 4


def test_flat_field_generator_size_and_values():
    flat_field_source=FlatFieldGenerator(test_star_field_size_x, 
                                         test_star_field_size_y, test_flat_photon_count)
    flat_photon_field = flat_field_source.generate_psf()

    # Test size and flat values
    assert np.sum(flat_photon_field) == test_star_field_size_x * test_star_field_size_x
    assert np.max(flat_photon_field) == test_flat_photon_count
    assert np.min(flat_photon_field) == test_flat_photon_count


def test_moffat_generator_psf_is_pdf():
    moffat_field_source=MoffatPSFGenerator(test_star_field_size_x, 
                                           test_star_field_size_x, test_alpha, test_beta)
    moffat_psf = moffat_field_source.generate_psf()

    # Test a PDF
    assert np.sum(moffat_psf) < (1.0 + pdf_sum_tolerance) 
    assert np.sum(moffat_psf) > (1.0 - pdf_sum_tolerance) 

    # Test Max Value at center
    result = np.where(moffat_psf == np.amax(moffat_psf))
    list_of_coordinates = list(zip(result[0], result[1]))
    for coordinate in list_of_coordinates:
        assert coordinate[0] == test_star_field_size_x/2
        assert coordinate[1] == test_star_field_size_y/2

    # Test Min Value at a corner
    result = np.where(moffat_psf == np.amin(moffat_psf))
    list_of_coordinates = list(zip(result[0], result[1]))
    for coordinate in list_of_coordinates:
        assert coordinate[0] == 0
        assert coordinate[1] == 0


def test_gaussian_generator_psf_is_pdf():
    gaussian_field_source=GaussianPSFGenerator(test_star_field_size_x, 
                                           test_star_field_size_x, test_sigma_x, test_sigma_x)
    gaussian_psf = gaussian_field_source.generate_psf()

    # Test a PDF
    assert np.sum(gaussian_psf) < (1.0 + pdf_sum_tolerance) 
    assert np.sum(gaussian_psf) > (1.0 - pdf_sum_tolerance) 

    # Test Max Value at center
    result = np.where(gaussian_psf == np.amax(gaussian_psf))
    list_of_coordinates = list(zip(result[0], result[1]))
    for coordinate in list_of_coordinates:
        assert coordinate[0] == test_star_field_size_x/2
        assert coordinate[1] == test_star_field_size_y/2

    # Test Min Value at a corner
    result = np.where(gaussian_psf == np.amin(gaussian_psf))
    list_of_coordinates = list(zip(result[0], result[1]))
    for coordinate in list_of_coordinates:
        assert coordinate[0] == 0
        assert coordinate[1] == 0


def test_diffraction_generator_psf_is_pdf():
    diffraction_source=DiffractionPSFGenerator(test_n_screen_size, 
                              test_padding_factor)
    diffraction_psf = diffraction_source.generate_psf()

    # Test a PDF
    assert np.sum(diffraction_psf) < (1.0 + pdf_sum_tolerance) 
    assert np.sum(diffraction_psf) > (1.0 - pdf_sum_tolerance) 

    # Test Max Value at center
    result = np.where(diffraction_psf == np.amax(diffraction_psf))
    list_of_coordinates = list(zip(result[0], result[1]))
    for coordinate in list_of_coordinates:
        assert coordinate[0] == 1024/2
        assert coordinate[1] == 1024/2


def test_seeing_generator_psf_is_pdf():
    seeing_source=SeeingPSFGenerator(test_n_screen_size, test_telescope_diameter, 
                              test_pixel_scale, test_r0, test_L0, 
                              test_stencil_length_factor, test_wavelength_nm,
                              test_padding_factor)
    seeing_psf = seeing_source.generate_psf()

    # Test a PDF
    assert np.sum(seeing_psf) < (1.0 + pdf_sum_tolerance) 
    assert np.sum(seeing_psf) > (1.0 - pdf_sum_tolerance) 

def test_seeing_generator_shift():
    seeing_source=SeeingPSFGenerator(test_n_screen_size, test_telescope_diameter, 
                              test_pixel_scale, test_r0, test_L0, 
                              test_stencil_length_factor, test_wavelength_nm,
                              test_padding_factor)
    seeing_psf_1 = seeing_source.generate_psf()
    seeing_psf_2 = seeing_source.generate_psf()

    # Test sequential PSF not equal
    assert np.array_equal(seeing_psf_1, seeing_psf_2) == False

def test_source_photon_generator_poisson_sampling():
    flat_field_source=FlatFieldGenerator(test_star_field_size_x, 
                                         test_star_field_size_y, test_flat_photon_count)
    flat_field_sampled = flat_field_source.photon_sample()

    # Test photon sampling of flat field within 20% of expected value (1000 samples)
    expected_value = np.sum(flat_field_sampled)/(test_star_field_size_x*test_star_field_size_y)

    assert expected_value < (1.0 + test_poisson_mean_tolerance) 
    assert expected_value > (1.0 - test_poisson_mean_tolerance) 

    # Test photon sampling changes each time
    flat_field_sampled_1 = flat_field_source.photon_sample()
    flat_field_sampled_2 = flat_field_source.photon_sample()

    # Test sequential PSF not equal
    assert np.array_equal(flat_field_sampled_1, flat_field_sampled_2) == False

def test_source_photon_generator_poisson_sampling_scaled():
    gaussian_field_source=GaussianPSFGenerator(test_star_field_size_x, 
                                           test_star_field_size_x, test_sigma_x, test_sigma_x)
    gaussian_scaled_sampled = gaussian_field_source.scaled_photon_sample(test_total_photon)

    generated_photon_count = np.sum(gaussian_scaled_sampled)

    # Test photon counted generated over entire PSF within tolerance of target
    assert generated_photon_count < (1.0 + photon_count_tolerance) * test_total_photon
    assert generated_photon_count > (1.0 - photon_count_tolerance) * test_total_photon

def test_scaled_photon_sample_stack():
    gaussian_field_source=GaussianPSFGenerator(test_star_field_size_x, 
                                           test_star_field_size_x, test_sigma_x, test_sigma_x)
    input_image_stack=gaussian_field_source\
        .scaled_photon_sample_stack(test_total_photon, test_frame_count)

    # Test correct number of frames generated
    assert input_image_stack.shape[0] == test_frame_count

    # Test different frames are unique
    assert np.array_equal(input_image_stack[0, :, :], input_image_stack[1, :, :]) == False




