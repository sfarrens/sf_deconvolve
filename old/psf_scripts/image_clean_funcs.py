import numpy as np
from scipy.linalg import norm
from functions import stats
from psf.noise import denoise, get_auto_correl_map
from psf.image_file_io import read_fits_image, gen_data_cube

path = '/Users/sfarrens/Data/Work/CEA/Fred/Galaxy/galaxy/'
fake_image = path + 'fake_output.mr'
wave_image = path + 'real_output.mr'

fake_data_norm = norm(read_fits_image(fake_image)[0], 2)
wave_data = read_fits_image(wave_image)[0]


def gen_wave_cube(fits_data, pixel_rad, n_obj=None):
    return gen_data_cube(wave_data, fits_data, pixel_rad, n_obj=n_obj)


def get_sigma(data):
    return stats.sigma_mad(data) / fake_data_norm


def clean_image(data, sigma, window_rad=1, k=4):
    map = get_auto_correl_map(data, window_rad)
    return denoise(data, (1 - map) * k * sigma, 'soft')


def clean_all(data_cube, wave_cube):
    return np.array([clean_image(data, get_sigma(wave)) for data, wave
                     in zip(data_cube, wave_cube)])
