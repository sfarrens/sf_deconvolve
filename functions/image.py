#  @file image.py
#
#  IMAGE ANALYSIS FUNCTIONS
#
#  Basic functions for analysing images and/or 2D-arrays.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from collections import deque
from scipy.ndimage.interpolation import shift
from itertools import product, izip
from functions.np_adjust import data2np, pad2d


##
#  Function that convolves and image with a kernal using FFT.
#
#  @param[in] image: 2D image data.
#  @param[in] kernel: 2D kernel.
#
#  @return Convolved image.
#
def fftconvolve(image, kernel):

    x = np.fft.fftshift(np.fft.fftn(image))
    y = np.fft.fftshift(np.fft.fftn(kernel))

    return np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x * y))))


##
#  Function that deconvolves and image with a kernal using FFT.
#
#  @param[in] image: 2D image data.
#  @param[in] kernel: 2D kernel.
#
#  @return Deconvolved image.
#
def fftdeconvolve(image, kernel):

    x = np.fft.fftshift(np.fft.fftn(image))
    y = np.fft.fftshift(np.fft.fftn(kernel))

    return np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x / y))))


##
#  Function that downsamples (decimates) an image.
#
#  @param[in] image: 2D image data.
#  @param[in] factor: Downsampling factor.
#
#  @return Downsampled image.
#
#  @exception ValueError for invalid downsampling factor values.
#  @exception ValueError for invalid number of downsampling factors.
#
def downsample(image, factor):

    factor = np.array(factor)

    if not np.all(factor > 0):
        raise ValueError('The downsampling factor values must be > 0.')

    if factor.size == 1:
        return image[0::factor, 0::factor]

    elif factor.size == 2:
        return image[0::factor[0], 0::factor[1]]

    else:
        raise ValueError('The downsampling factor can only contain one or ' +
                         'two values.')


##
#  Function that returns an image with odd dimensions.
#
#  @param[in] image: 2D image data.
#
#  @return Uneven shaped image.
#
def resize_even_image(image):

    return image[[slice(x) for x in (np.array(image.shape) +
                  np.array(image.shape) % 2 - 1)]]


##
#  Function that finds the centres of the individual images in a 2D map.
#
#  @param[in] data_size: Size of 2D data map.
#  @param[in] layout: 2D Layout of image map.
#
#  @return Array of image centres.
#
def image_centres(data_size, layout):

    data_size = np.array(data_size)
    layout = np.array(layout)

    if data_size.size != 2:
        raise ValueError('The the data size must be of size 2.')
    if layout.size != 2:
        raise ValueError('The the layout must be of size 2.')

    ranges = np.array(list(product(*np.array([np.arange(x) for x in layout]))))
    image_size = data_size / layout
    image_centre = image_size / 2

    return image_centre + image_size * ranges


##
#  Function that rolls an array in 2 dimensions.
#
#  @param[in] data: 2D Input data.
#  @param[in] roll_rad: Roll radius in each dimension.
#
#  @return Rolled array.
#
def roll_2d(data, roll_rad=(1, 1)):

    return np.roll(np.roll(data, roll_rad[1], axis=1), roll_rad[0], axis=0)


##
#  Function that rotates (by 180 deg) and rolls a 2D array.
#
#  @param[in] data: 2D Input data.
#
#  @return Rotated and rolled array.
#
def rot_and_roll(data):

    return roll_2d(np.rot90(data, 2), -(np.array(data.shape) / 2))


##
#  Function that generates an image mask.
#
#  @param[in] kernel_shape: 2D shape of the kernel.
#  @param[in] image_shape: 2D shape of the image.
#
#  @return Boolean mask.
#
def gen_mask(kernel_shape, image_shape):

    kernel_shape = np.array(kernel_shape)
    image_shape = np.array(image_shape)

    shape_diff = image_shape - kernel_shape

    mask = np.ones(image_shape, dtype=bool)

    if shape_diff[0] > 0:
        mask[-shape_diff[0]:] = False
    if shape_diff[1] > 0:
        mask[:, -shape_diff[1]:] = False

    return roll_2d(mask, -(kernel_shape / 2))


##
#  Function that generates the roll sequence for a 2D array.
#
#  @param[in] data_shape: 2D shape of the data.
#
#  @return List of roll radii.
#
def roll_sequence(data_shape):

    data_shape = np.array(data_shape)

    return list(product(*(range(data_shape[0]), range(data_shape[1]))))


##
#  Function that generates the kernel pattern. Rather than padding the kernel
#  with zeroes to match the image size one simply extracts the series of
#  repitions of the base kernel patterns.
#
#  @param[in] kernel_shape: 2D shape of the kernel.
#  @param[in] mask: The image mask.
#
#  @return List of roll radii.
#
def kernel_pattern(kernel_shape, mask):

    kernel_shape = np.array(kernel_shape)

    kernel_buffer = 1 - np.array(kernel_shape) % 2

    n_rep_axis1 = sum(1 - mask[:, 0])
    n_rep_axis2 = sum(1 - mask[0])

    if np.any(mask[:, 0] is False):
        pos_1 = np.where(mask[:, 0] is False)[0][0] - 1 + kernel_buffer[0]

    if np.any(mask[0] is False):
        pos_2 = np.where(mask[0] is False)[0][0] - 1 + kernel_buffer[1]

    pattern = np.arange(np.prod(kernel_shape)).reshape(kernel_shape)

    for i in range(n_rep_axis1):
        pattern = np.insert(pattern, pos_1, pattern[pos_1], axis=0)

    for i in range(n_rep_axis2):
        pattern = np.insert(pattern, pos_2, pattern[:, pos_2], axis=1)

    return pattern.reshape(pattern.size)


##
#  Function that rearanges the input kernel elements for vector multiplication.
#  The input kernel is padded with zeroes to match the image size.
#
#  @param[in] kernel: 2D Input kernel.
#  @param[in] data_shape: 2D data shape.
#
#  @return Rearanged matrix of kernel elements.
#
def rearrange_kernel(kernel, data_shape=None):

    # Define kernel shape.
    kernel_shape = np.array(kernel.shape)

    # Set data shape if not provided.
    if isinstance(data_shape, type(None)):
        data_shape = kernel_shape
    else:
        data_shape = np.array(data_shape)

    # Set the length of the output matrix rows.
    vec_length = np.prod(data_shape)

    # Find the diffrence between the shape of the data and the kernel.
    shape_diff = data_shape - kernel_shape

    if np.any(shape_diff < 0):
        raise ValueError('Kernel shape must be less than or equal to the '
                         'data shape')

    # Set the kernel radius.
    kernel_rad = kernel_shape / 2

    # Rotate, pad and roll the input kernel.
    kernel_rot = np.pad(np.rot90(kernel, 2), ((0, shape_diff[0]),
                        (0, shape_diff[1])), 'constant')
    kernel_rot = np.roll(np.roll(kernel_rot, -kernel_rad[1], axis=1),
                         -kernel_rad[0], axis=0)

    return np.array([np.roll(np.roll(kernel_rot, i, axis=0), j,
                    axis=1).reshape(vec_length) for i in range(data_shape[0])
                    for j in range(data_shape[1])])


##
#  Function that selects a window of a given size from a 2D-array.
#
#  Note that the image edges are padded with zeros.
#
#  @param[in] data: 2D Input array.
#  @param[in] pos: 2D Position in array where the window is to be centred.
#  @param[in] pixel_rad: 1D or 2D pixel radius.
#
#  @return Windowed selection from input array.
#
def window(data, pos, pixel_rad):

    # Check that the input array has two dimensions.
    data = np.array(data)
    if data.ndim != 2:
        raise ValueError('The input array must be 2D.')

    # If the pixel radius size is one repat the value for the 2nd dimension.
    pixel_rad = np.array(pixel_rad)
    if pixel_rad.size == 1:
        pixel_rad = np.repeat(pixel_rad, 2)
    if pixel_rad.size not in (1, 2):
        raise ValueError('The pixel radius must have a size of 1 or 2.')

    # Check that the array position has a size of two.
    pos = np.array(pos) % np.array(data.shape)
    if pos.size != 2:
        raise ValueError('The array position must have a size of 2.')

    # Check if the pixel radius is within the bounds of the input array.
    if (np.any(pixel_rad < 1) or np.any(np.array(data.shape) / 2 -
                                        pixel_rad < 0)):
        raise ValueError('The pixel radius values must have a value of at '
                         'least 1 and at most half the size of the input '
                         'array. Array size: ' + str(data.shape))

    # Return window selection.
    return pad2d(data, pixel_rad)[[slice(a, a + 2 * b + 1) for a, b in
                                   zip(pos, pixel_rad)]]


##
#  Function that returns all of the pixel positions from a 2D-array.
#
#  @param[in] array_shape: Shape of 2D-array.
#
#  @return List of pixel positions.
#
def pixel_pos(array_shape):

    ranges = np.array([np.arange(x) for x in np.array(array_shape)])

    return list(product(*ranges))


##
#  Class that returns window selections of a given pixel radius from an input
#  2D-array.
#
class FetchWindows():

    ##
    #  Method that initialises the class instance.
    #
    #  @param[in] data: 2D Input array.
    #  @param[in] pixel_rad: Pixel radius.
    #  @param[in] all: Select all pixels.
    #
    def __init__(self, data, pixel_rad, all=False):
        self.data = np.array(data)
        self.shape = np.array(self.data.shape)
        self.pixel_rad = np.array(pixel_rad)
        self._check_inputs()
        self._pad_data()
        if all:
            self.n_pixels()

    ##
    #  Method that checks the input values.
    #
    #  @exception ValueError for invalid number of dimensions of input array.
    #  @exception ValueError for invalid size of pixel radius.
    #
    def _check_inputs(self):
        if self.data.ndim != 2:
            raise ValueError('The input array must be 2D.')
        if self.pixel_rad.size == 1:
            self.pixel_rad = np.repeat(self.pixel_rad, 2)
        elif self.pixel_rad.size not in (1, 2):
            raise ValueError('The pixel radius must have a size of 1 or 2.')

    ##
    #  Method that pads the input array with zeros.
    #
    def _pad_data(self):
        self.pad_data = pad2d(self.data, self.pixel_rad)

    ##
    #  Method that adjusts the pixel positions according to the pixel radius.
    #
    def _adjust_pixels(self):
        self.pixels = self.pixels % self.shape + self.pixel_rad

    ##
    #  Method that gets the desired pixel positions.
    #
    #  @param[in] pixels: Pixel positions.
    #
    #  @exception ValueError for invalid number of dimensions of pixel
    #  positions.
    #  @exception ValueError for invalid size of second dimension of pixel
    #  positions.
    #
    def get_pixels(self, pixels):
        self.pixels = np.array(pixels)
        if not 1 <= self.pixels.ndim <= 2:
            raise ValueError('Invalid number of dimensions for pixels')
        elif self.pixels.ndim == 2 and self.pixels.shape[1] != 2:
            raise ValueError('The second dimension of pixels must have size 2')
        self._adjust_pixels()

    ##
    #  Method that gets all pixel positions.
    #
    #  @param[in] n_pixels: Number of pixels to keep. [Default is None]
    #  @param[in] random: Shuffle the pixel positions. [Default is False]
    #
    #  Note: When n_pixels=None all pixels are kept.
    #
    def n_pixels(self, n_pixels=None, random=False):
        self.pixels = pixel_pos(self.shape)
        if random:
            np.random.shuffle(self.pixels)
        self.pixels = self.pixels[:n_pixels]
        self._adjust_pixels()

    ##
    #  Method that retrieves a window from the padded input data at a given
    #  position.
    #
    #  @param[in] pos: Pixel position in 2D padded array.
    #  @param[in] func: Optional function to be applied to window selection.
    #
    #  @exception ValueError for invalid size of pixel position.
    #
    def _window(self, pos, func=None, *args):
        pos = data2np(pos)
        if pos.size != 2:
            raise ValueError('The pixel position must have a size of 2.')
        window = self.pad_data[[slice(a - b, a + b + 1) for a, b in
                                izip(pos, self.pixel_rad)]]
        if isinstance(func, type(None)):
            return window
        else:
            return func(window, *args)

    ##
    #  Method that scans the 2D padded input array and retrieves the windows
    #  at all the desired pixel positions.
    #
    #  @param[in] func: Optional function to be applied to window selection.
    #  @param[in] *args: Function arguments.
    #  @param[in] **kwargs: Keyword arguments.
    #
    def scan(self, func=None, *args, **kwargs):

        if 'arg_type' in kwargs and kwargs['arg_type'] == 'list':
            return np.array([self._window(pos, func, arg) for pos, arg in
                             zip(self.pixels, *args)])

        else:
            return np.array([self._window(pos, func, *args) for pos in
                             self.pixels])


##
#  Class that produces a Summed Area Table (SAT) for fast and efficient
#  statistics on image patches.
#
#  Note: Also know as Itegral Image (i in the class features).
#
class SAT():

    ##
    #  Method that initialises the class instance.
    #
    #  @param[in] data: 2D Input array.
    #
    def __init__(self, data):

        self.x = data
        self.x2 = self.x ** 2
        self.i = self.x.cumsum(axis=0).cumsum(axis=1)
        self.i2 = self.x2.cumsum(axis=0).cumsum(axis=1)
        self.i_pad = pad2d(self.i, 1)
        self.i2_pad = pad2d(self.i2, 1)

    ##
    #  Method calculates the area of a patch.
    #
    #  @param[in] data: 2D Input array.
    #  @param[in] corners: Positions of upper left and bottom right corners.
    #
    def get_area(self, data, corners):

        corners = np.array(corners)
        corners[1] += 1

        a = data[zip(corners[0])]
        b = data[corners[0, 0], corners[1, 1]]
        c = data[corners[1, 0], corners[0, 1]]
        d = data[zip(corners[1])]

        return float(a + d - b - c)

    ##
    #  Method calculates the number of pixels in a patch.
    #
    #  @param[in] corners: Positions of upper left and bottom right corners.
    #
    def get_npx(self, corners):

        return np.prod(np.diff(corners, axis=0) + 1)

    ##
    #  Method calculates the variance and standard deviation of a patch.
    #
    def get_var(self):

        self.var = (self.area2 - (self.area ** 2 / self.npx)) / self.npx
        self.std = np.sqrt(self.var)

    ##
    #  Method that sets the corner positions of a single patch.
    #
    #  @param[in] corners: Positions of upper left and bottom right corners.
    #
    def set_patch(self, corners):

        self.area = self.get_area(self.i_pad, corners)
        self.area2 = self.get_area(self.i2_pad, corners)
        self.npx = self.get_npx(corners)
        self.get_var()

    ##
    #  Method that sets the corner positions for multiple patches.
    #
    #  @param[in] corners: List of the positions of upper left and bottom
    #  right corners.
    #
    def set_patches(self, corners):

        self.area = np.array([self.get_area(self.i_pad, corner)
                              for corner in corners])
        self.area2 = np.array([self.get_area(self.i2_pad, corner)
                               for corner in corners])
        self.npx = np.array([self.get_npx(corner) for corner in corners])
        self.get_var()
