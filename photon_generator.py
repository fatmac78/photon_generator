#  Class to generate photons for a simulated star - produces both 
#  diffraction limited psfd and seeing limited psfs
import aotools
import os
import sys
import numpy as np
from scipy.stats import poisson
import math 
import abc
from abc import ABC, abstractmethod
from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
from astropop.math.moffat import moffat_2d

class SourcePhotonGenerator(metaclass=abc.ABCMeta):
    """Abstract class that can generate a PSF from which photons may be sampled
    """

    def __init__(self):
        pass

    @abstractmethod
  
    def generate_psf(self, image: np.ndarray) -> np.ndarray:
        """ Generates a PSF, summing to 1
        """
        pass

    def get_photon(self, rate) -> int:
        value = poisson.rvs(mu=rate, size=1)
        return value[0]

    def photon_sample(self) -> np.ndarray:
        """ Generates a poisson sampled photon frame, given a flux map
        """
        flux_map = self.generate_psf()
        get_photon_vectorized = np.vectorize(self.get_photon)
        return get_photon_vectorized(flux_map)

    def scaled_photon_sample(self, total_photon: int) -> np.ndarray:
        """" Generates a poisson sampled photon frame 
        """
        flux_map = self.generate_psf()
        scaled_flux_map = flux_map * total_photon
        get_photon_vectorized = np.vectorize(self.get_photon)
        return get_photon_vectorized(scaled_flux_map)
    
    def scaled_photon_sample_stack(self, total_photon: int, frames: int) -> np.ndarray:
        """" Generates a poisson sampled photon frame stack, given a flux map 
        and total photon count
        """
        stack = []
        for frame in range(0, frames):
            stack.append(self.scaled_photon_sample(total_photon))
        stack = np.stack(stack, axis=0)
        return stack

class FlatFieldGenerator(SourcePhotonGenerator):
    """Generates a flat field, where each pixel has the specfied photon count value
    """

    def __init__(self, star_field_size_x, star_field_size_y, photon_count):
        self.star_field_size_x = star_field_size_x
        self.star_field_size_y = star_field_size_y
        self.photon_count = photon_count
        
    def generate_psf(self):
        flat_psf = np.full((self.star_field_size_x, self.star_field_size_y), self.photon_count)
        self.psf = flat_psf
        return flat_psf

class MoffatPSFGenerator(SourcePhotonGenerator):
    """Generates a Moffat shaped PSF which sums to 1 
    """

    def __init__(self, star_field_size_x, star_field_size_y, alpha, beta):
        self.star_field_size_x = star_field_size_x
        self.star_field_size_y = star_field_size_y
        self.alpha = alpha
        self.beta = beta
        
    def generate_psf(self):
        assert isinstance(self.star_field_size_x, int)
        assert isinstance(self.star_field_size_y, int)
        x0 = self.star_field_size_x // 2
        y0 = self.star_field_size_y // 2
        x = np.arange(0, self.star_field_size_x, dtype=float)
        y = np.arange(0, self.star_field_size_y, dtype=float)[:, np.newaxis]
        x -= x0
        y -= y0
        psf=moffat_2d(x, y, 0, 0, self.alpha, self.beta, 1, 0)
        return psf/np.sum(psf)

class GaussianPSFGenerator(SourcePhotonGenerator):
    """Generates a Gaussian shaped PSF which sums to 1 
    """

    def __init__(self, star_field_size_x, star_field_size_y, sigma_x, sigma_y):
        super().__init__()
        self.star_field_size_x = star_field_size_x
        self.star_field_size_y = star_field_size_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def generate_psf(self):
        assert isinstance(self.star_field_size_x, int)
        assert isinstance(self.star_field_size_y, int)
        x0 = self.star_field_size_x // 2
        y0 = self.star_field_size_y // 2
        x = np.arange(0, self.star_field_size_x, dtype=float)
        y = np.arange(0, self.star_field_size_y, dtype=float)[:, np.newaxis]
        x -= x0
        y -= y0
        exp_part = x**2/(2*self.sigma_x**2)+ y**2/(2*self.sigma_y**2)
        psf = 1/(2*np.pi*self.sigma_x*self.sigma_y) * np.exp(-exp_part)
        return psf/np.sum(psf)


class SeeingPSFGenerator(SourcePhotonGenerator):
    """Generates a Speckle PSF which sums to 1 using a Komolgorov phase-screen
    """

    def __init__(self, n_screen_size, telescope_diameter, pixel_scale,
                 r0, L0, stencil_length_factor, wavelength_nm, padding_factor):
        super().__init__()
        self.n_screen_size = n_screen_size
        self.telescope_diameter = telescope_diameter
        self.pixel_scale = telescope_diameter/n_screen_size
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = stencil_length_factor
        self.wavelength = wavelength_nm
        self.padding_factor = padding_factor

        # the screen should be instatiated once here and then shifted each time generated psf called

        self.phase_screen_nm = PhaseScreenKolmogorov(self.n_screen_size, 
                                                self.pixel_scale, 
                                                self.r0, self.L0, 
                                                self.stencil_length_factor)
    def generate_psf(self):
    
        # Shift phase screen one row each time it is called
        self.phase_screen_nm.add_row()

        # Convert phase screen from nm to radians
        phase_screen_radians = \
            (self.phase_screen_nm.scrn * 2 * math.pi) / self.wavelength

        # Generate pupil mask
        pupil_mask = aotools.circle(self.n_screen_size/2, self.n_screen_size)\
             - aotools.circle(self.n_screen_size/8, self.n_screen_size)

        # Multiply pupil by phase 
        phase = np.exp(phase_screen_radians*(0.+1.j))
        distorted_wavefront = pupil_mask * phase

        # Create padding to get an oversampled psf to make it look nice
        padded_pupil = np.zeros((self.n_screen_size*self.padding_factor, 
                                 self.n_screen_size*self.padding_factor))
        padded_pupil[:self.n_screen_size, :self.n_screen_size] \
             = distorted_wavefront
    
        # Transform from the pupil to the focal plane
        ft = aotools.ft2(padded_pupil, delta=1./self.n_screen_size)
        seeing_psf = np.real(np.conj(ft)*ft)

        return seeing_psf/np.sum(seeing_psf)


class DiffractionPSFGenerator(SourcePhotonGenerator):
    """Generates a Diffraction-limited PSF which sums to 1 
    """

    def __init__(self, n_screen_size, padding_factor):
        super().__init__()
        self.n_screen_size = n_screen_size
        self.padding_factor = padding_factor
       
    def generate_psf(self):
    
        # Generate pupil mask
        pupil_mask = aotools.circle(self.n_screen_size/2, self.n_screen_size)\
             - aotools.circle(self.n_screen_size/8, self.n_screen_size)

        # Create padding to get an oversampled psf to make it look nice
        padded_pupil = np.zeros((self.n_screen_size*self.padding_factor, 
                                 self.n_screen_size*self.padding_factor))
        padded_pupil[:self.n_screen_size, :self.n_screen_size] \
             = pupil_mask
    
        # Transform from the pupil to the focal plane
        ft = aotools.ft2(padded_pupil, delta=1./self.n_screen_size)
        powft = np.real(np.conj(ft)*ft)
        diffraction_psf =  powft    

        return diffraction_psf/np.sum(diffraction_psf)