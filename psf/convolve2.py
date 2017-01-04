from functions.image import FetchWindows


##
#  Function that convolves an image with a pixel variant PSF.
#
#  @param[in] image: Input image data.
#  @param[in] psf: Pixel variant PSF.
#
#  @return Convolved image.
#
def psf_var_convolve(image, psf):

    def get_convolve(sub_image, psf):
        return np.sum(sub_image * np.rot90(psf, 2))

    w = FetchWindows(image, psf.shape[-1] / 2, all=True)

    return w.scan(get_convolve, psf, arg_type='list').reshape(image.shape)


##
#  Function that convolves the input data with the principal components of a
#  PSF.
#
#  @param[in] data: Input data.
#  @param[in] psf_pcs: PSF principal components.
#  @param[in] psf_coef: PSF coefficients.
#  @param[in] pcs_rot: Option to rotate principal components.
#
#  @return Convolved data.
#
def pca_convolve(data, psf_pcs, psf_coef, pcs_rot=False):

    if pcs_rot:
        return sum((convolve(data * b, rotate(a)) for a, b in
                   zip(psf_pcs, psf_coef)))

    else:
        return sum(((convolve(data, a) * b) for a, b in
                   zip(psf_pcs, psf_coef)))


##
#  Function that convolves the input data stack with the principal components
#  of a PSF.
#
#  @param[in] data_stack: Input data stack.
#  @param[in] psf_pcs: PSF principal components.
#  @param[in] psf_coef_stack: Stack of PSF coefficients.
#  @param[in] pcs_rot: Option to rotate principal components.
#
#  @return Convolved data stack.
#
def pca_convolve_stack(data_stack, psf_pcs, psf_coef_stack, pcs_rot=False):

    return np.array([pca_convolve(data, psf_pcs, psf_coef, pcs_rot) for
                     data, psf_coef in zip(data_stack, psf_coef_stack)])
