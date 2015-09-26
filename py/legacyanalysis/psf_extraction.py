#!/usr/env python

import numpy as np
import scipy.ndimage

from scipy.special import eval_chebyt

import astropy.wcs
import astropy.io
import astropy.convolution
import astropy.modeling

import warnings

from sklearn.neighbors import NearestNeighbors

try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits


def fit_2d_gaussian(img):
    '''
    Fit a 2D Gaussian to an image. The 2D Gaussian model includes a center
    (x, y), a standard deviation along each axis, a rotation angle and an
    overall amplitude.

    Input:
      img  A 2D numpy array representing the image to be fit.

    Output:
      model_fit  An astropy.modeling.models.Gaussian2D object. Calling
                 `model_fit(x,y)` will return the value of the model Gaussian
                 at (x, y). See the astropy documentation for more details on
                 how to interact with this class.
    '''

    shape = img.shape
    y, x = np.mgrid[:shape[1], :shape[0]]

    # Fit a 2D Gaussian to the data
    model_guess = astropy.modeling.models.Gaussian2D(
        x_mean=((shape[0]-1.)/2.),
        y_mean=((shape[1]-1.)/2.),
        x_stddev=1.,
        y_stddev=1.,
        amplitude=np.max(img)
    )

    fitter = astropy.modeling.fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('default')
        # TODO: bound the model parameters to reasonable values.
        model_fit = fitter(model_guess, x, y, img)

    return model_fit


def img_center_of_mass(img, img_mask):
    '''
    Find the center of mass of an image.
    '''

    idx = ~np.isfinite(img) | (img_mask != 0) # Determine image mask
    idx |= idx[:,::-1] | idx[::-1,:] # Make mask symmetric under parity flips
    img_clean = np.empty(img.shape, dtype='f8')
    img_clean[:] = img[:]
    img_clean[idx] = 0.

    return scipy.ndimage.measurements.center_of_mass(img_clean)


def gen_centered_weights(shape, sigma, n_sigma_clip=3.):
    '''
    Return a weight image that weights pixels towards the center of the image
    more strongly than pixels away from the center.
    '''
    w = np.array(astropy.convolution.Gaussian2DKernel(
        sigma, x_size=shape[0], y_size=shape[1],
        mode='oversample', factor=5
    ))

    w /= np.max(w)
    w[w < np.exp(-0.5 * n_sigma_clip**2.)] = 0.

    return w


def recenter_postage_stamps(exposure_img, weight_img, mask_img, star_x, star_y,
                            ps_exposure, ps_weight, ps_mask, **kwargs):
    '''
    Recenter a stack of postage stamps, trying to properly centroid each star.
    Should work for small shifts (< 1 pixel). Calculates a weighted center of
    mass for each postage stamp (the weighting ignores pixels away from the
    center of the image), and then goes back into the original exposure image
    and extracts the stars again.
    '''

    max_shift = kwargs.pop('max_shift', 2.)

    # Weight to apply to each image before calculating center of mass
    w_cent = gen_centered_weights(ps_exposure.shape[1:],
                                  2.*max_shift,
                                  n_sigma_clip=1.)

    dxy = []
    star_x_cent, star_y_cent = [], []

    # Calculate the weighted center of mass of each postage stamp
    for ps_img, ps_mask in zip(ps_exposure, ps_mask):
        dxy.append(img_center_of_mass(ps_img*w_cent, ps_mask))

    dxy = np.array(dxy)
    dxy[:,0] -= 0.5 * float(ps_exposure.shape[1]-1.)
    dxy[:,1] -= 0.5 * float(ps_exposure.shape[2]-1.)

    # Find the stars that have only small shifts
    idx_good = np.all((np.abs(dxy) < max_shift) & np.isfinite(dxy), axis=1)

    # Apply the small shifts to the stellar positions
    star_x_cent = np.array(star_x)
    star_y_cent = np.array(star_y)
    star_x_cent[idx_good] += dxy[idx_good, 0]
    star_y_cent[idx_good] += dxy[idx_good, 1]

    ret = extract_stars(exposure_img, weight_img, mask_img,
                        star_x_cent, star_y_cent, **kwargs)

    return ret


def sinc_shift_image(img, dx, dy, roll_int=True):
    '''
    Shift a real-valued image by (dx,dy), using the DFT shift theorem.

    Input:
      img  A 2D image, represented by a 2D, real-valued numpy array.
      dx   The shift along the x-axis, in pixels
      dy   The shift along the y-axis, in pixels

    Output:
      A shifted copy of the image, as a 2D numpy array.
    '''

    # Roll the image first by the nearest integer amount
    if roll_int:
        dx_int = int(np.around(dx))
        dy_int = int(np.around(dy))

        dx = dx - dx_int
        dy = dy - dy_int

        img = np.roll(img, dx_int, axis=0)
        img = np.roll(img, dy_int, axis=1)

    # Use a sinc-shift for the rest of the shift
    img_dft = np.fft.fft2(img)
    img_dft = scipy.ndimage.fourier.fourier_shift(img_dft, (dx,dy))

    return np.real(np.fft.ifft2(img_dft))


def fit_star_offset(psf_coeffs, ps_exposure, ps_weight, ps_mask,
                    star_x, star_y, star_flux, sky_level, ccd_shape,
                    max_shift=5., verbose=False):
    #print psf_coeffs.shape
    #print star_x.shape

    # Calculate the local psf (if the shifts are small, this is constant)
    psf_img = eval_psf(psf_coeffs, star_x, star_y, ccd_shape)
    psf_img /= np.sum(psf_img)

    # Calculate the model image of the star
    star_model_img = psf_img * star_flux + sky_level

    # For constrained optimization, fit in a transformed parameter space. The
    # transformed parameter is given by
    #   x' = Dx * tan(pi/2 * x/x_max),
    # where Dx is a scale parameter. This is done so that for x' in (-inf, inf),
    # x is in (-x_max, x_max).

    Dx = 0.5 * max_shift

    # x' -> x
    def xp2x(xp):
        return max_shift * 2./np.pi * np.arctan(xp / Dx)

    # x -> x'
    def x2xp(x):
        return Dx * np.tan(0.5*np.pi * x / max_shift)

    # The objective function to be minimized:
    # chi^2/dof for arbitrary shift of star model image
    def f_obj(dxy_p):
        # Go from transformed parameter values to actual values
        dx, dy = xp2x(dxy_p)

        # Shift the model image of the star (negative b/c we're shifting the
        # model, rather than the actual image of the star)
        model_shifted = sinc_shift_image(star_model_img, dx, dy)

        # chi^2/dof vs. actual image of star
        resid = ps_exposure - model_shifted
        resid *= resid * ps_weight
        resid[(ps_mask != 0) | ~np.isfinite(resid)] = 0.
        chisq_dof = np.sum(np.sum(resid, axis=0), axis=0)
        chisq_dof /= float(ps_exposure.shape[0] * ps_exposure.shape[1])

        # Add a penalty for wandering very far off
        chisq_dof += 0.1 * np.sum(dxy_p**2.) / Dx**2.

        #print dx, dy, chisq_dof

        return chisq_dof

    # Minimize chi^2/dof by shifting the model image of the star
    # TODO: Test other algorithms, like 'BFGS', and see why some stars only use one iteration.
    x0 = np.array([0.,0.])
    res = scipy.optimize.minimize(f_obj, x0, method='BFGS',#'nelder-mead',
                                  options={'xtol': 1.e-8, 'disp': verbose, 'maxiter': 250})

    # Transform the result from primed to unprimed coordinates
    dx_opt, dy_opt = xp2x(res.x)

    return dx_opt, dy_opt


def gen_stellar_flux_predictor(star_flux, star_ps1_mag,
                               star_x, star_y, ccd_shape,
                               psf_coeffs):
    n_stars = star_flux.size
    #print 'CCD Shape:', ccd_shape

    # Calculate corrections to fluxes, by summing the PSF at the location of
    # each star
    # TODO: Verify that no normalization is needed, and remove this code
    psf_norm = np.ones(n_stars, dtype='f8')#np.empty(n_stars, dtype='f8')

    #for k in range(n_stars):
    #    # Evaluate the PSF at the location of the star
    #    psf_model = eval_psf(psf_coeffs, star_x[k], star_y[k], ccd_shape)
    #    psf_norm[k] = np.sum(psf_model)

    #print 'PSF Norm:', psf_norm

    star_flux_corr = star_flux * psf_norm

    # TODO: Automatic choice of bands to use

    # Convert PS1 magnitude to flux, where 20th mag corresponds to a flux of 1
    ps1_flux = 10.**(-(star_ps1_mag-20.) / 2.5)

    # The model is that the fitted flux is a linear combination of the fluxes
    # in the PS1 bands, plus a zero point flux. The design matrix is therefore
    # given by
    #
    #   A = [[1 g_0 r_0 i_0 z_0 y_0],
    #        [1 g_1 r_1 i_1 z_1 y_1],
    #                  ...
    #        [1 g_n r_n i_n z_n y_n]] ,
    #
    # where n is the number of stars, while the parameter matrix is given by
    #
    #   x = [f_zp c_g c_r c_i c_z c_y] ,
    #
    # and the data matrix is given by
    #
    #   b = [f_0 f_1 ... f_n] ,
    #
    # where f_k is the flux of star k.

    n_stars = star_flux.size

    A = np.empty((n_stars, 6), dtype='f8')
    A[:,0] = 1.
    A[:,1:] = ps1_flux[:]

    # Use only stars with 5-band photometry
    idx_good = np.all((star_ps1_mag > 1.) & (star_ps1_mag < 26.), axis=1)
    idx_good &= np.isfinite(star_flux)

    #print idx_good
    print '# of good stars: {}'.format(np.sum(idx_good))

    A = A[idx_good]
    b = star_flux_corr[idx_good]

    # Set errors in fitted fluxes at 5%
    #sigma = b * 0.05
    #sqrt_w = 1. / sigma
    #A *= sqrt_w[:,None]
    #b *= sqrt_w[:]

    # Remove NaN and Inf values
    A[~np.isfinite(A)] = 0.
    b[~np.isfinite(b)] = 0.

    # Execute least squares
    coeffs = np.linalg.lstsq(A, b)[0]
    zp = coeffs[0]
    c = coeffs[1:]

    #print 'Flux fit coefficients:'
    #print coeffs
    #print ''

    def PS1_mag_to_fit_flux(m_p1):
        f_p1 = 10.**(-(m_p1-20.) / 2.5)
        return zp + np.einsum('k,jk->j', c, f_p1)

    #print star_ps1_mag[idx_good][:3]
    f_resid = (PS1_mag_to_fit_flux(star_ps1_mag[idx_good]) - star_flux_corr[idx_good]) / star_flux_corr[idx_good]

    #print 'Residuals:'
    #print f_resid
    #print 'Residual percentiles:'
    #pctiles = [1., 5., 10., 20., 50., 80., 90., 95., 99.]
    #for p,r in zip(pctiles, np.percentile(f_resid, pctiles)):
    #    print '  {: 2d} -> {:.3f}'.format(int(p), r)
    #print ''

    return PS1_mag_to_fit_flux


def filter_close_points(x, y, r):
    '''
    Filter a list of points in 2D, flagging points that are too close to another
    point.

    Inputs:
      x  x-coordinates of points
      y  y-coordinates of points
      r  Radius at which to filter points

    Outputs:
      keep_idx  An boolean array, with True for points that have no near
                neighbor, and False for points that do.
    '''

    xy = np.vstack([x,y]).T
    nn = NearestNeighbors(radius=r)
    nn.fit(xy)
    idx_collection = nn.radius_neighbors(radius=r, return_distance=False)

    keep_idx = np.ones(x.size, dtype=np.bool)

    for idx in idx_collection:
        keep_idx[idx] = 0

    return keep_idx


def eval_psf(psf_coeffs, star_x, star_y, ccd_shape):
    '''
    Evaluate the PSF at one location on the CCD.

    Inputs:
      psf_coeffs  Images of the PSF at different orders. The shape is
                  (order, x, y). The zeroeth-order term is the term that is
                  constant over the CCD. Entries 1 and 2 describe the x- and
                  y-dependent terms, respectively. Entries 3 and 4 describe the
                  x^2- and y^2-dependent terms, respectively, while entry 5
                  describes the strength of the xy-dependent term.
      star_x      The x-coordinate (in pixels) at which to evaluate the PSF.
      star_y      The y-coordinate (in pixels) at which to evaluate the PSF.
      ccd_shape   The shape of the PSF, (n_x, n_y), in pixels.

    Output:
      psf_img  An image of the PSF evaluated at (star_x, star_y).
    '''

    x = 2. * star_x / float(ccd_shape[0]) - 1.
    y = 2. * star_y / float(ccd_shape[1]) - 1.

    psf_img = np.empty((psf_coeffs.shape[1], psf_coeffs.shape[2]), dtype='f8')

    # TODO: Generalize this to arbitrary orders of x,y
    psf_img[:,:] = psf_coeffs[0,:,:]
    psf_img[:,:] += psf_coeffs[1,:,:] * eval_chebyt(1, x)
    psf_img[:,:] += psf_coeffs[2,:,:] * eval_chebyt(1, y)
    psf_img[:,:] += psf_coeffs[3,:,:] * eval_chebyt(2, x)
    psf_img[:,:] += psf_coeffs[4,:,:] * eval_chebyt(2, y)
    psf_img[:,:] += psf_coeffs[5,:,:] * eval_chebyt(1, x) * eval_chebyt(1, y)

    return psf_img


def calc_psf_fluxes(psf_coeffs, star_x, star_y, ccd_shape):
    '''
    Calculate the total flux in the PSF (i.e., the sum over the PSF) at each of
    the specified locations on the CCD.

    Inputs:
      psf_coeffs  Images of the PSF at different orders. The shape is
                  (order, x, y). The zeroeth-order term is the term that is
                  constant over the CCD. Entries 1 and 2 describe the x- and
                  y-dependent terms, respectively. Entries 3 and 4 describe the
                  x^2- and y^2-dependent terms, respectively, while entry 5
                  describes the strength of the xy-dependent term.
      star_x      The x-coordinates (in pixels) at which to evaluate the PSF.
      star_y      The y-coordinates (in pixels) at which to evaluate the PSF.
      ccd_shape   The shape of the PSF, (n_x, n_y), in pixels.

    Output:
      psf_flux  The flux in the PSF at each location (same shape as `star_x`
                and `star_y`).
    '''
    return np.array([np.sum(eval_psf(psf_coeffs,x,y,ccd_shape)) for x,y in zip(star_x,star_y)])


def fit_star_params(psf_coeffs, star_x, star_y,
                    ps_exposure, ps_weight, ps_mask, ccd_shape,
                    sky_mean=0., sky_sigma=0.5,
                    stellar_flux_mean=0., stellar_flux_sigma=np.inf):
    '''
    Fit stellar flux and sky brightness, given a PSF model.
    '''

    # Evaluate the PSF at the location of the star
    psf_val = eval_psf(psf_coeffs, star_x, star_y, ccd_shape)
    psf_norm = np.sum(psf_val)
    psf_val /= psf_norm
    psf_val.shape = (psf_val.size,)

    # Normalize the stellar flux priors
    #psf_norm = np.sum(psf_val)
    #stellar_flux_mean = stellar_flux_mean / psf_norm
    #stellar_flux_sigma = stellar_flux_sigma / psf_norm

    # Calculate the square root of the weight
    sqrt_w = np.sqrt(ps_weight.flat)
    #sqrt_w = np.ones(ps_weight.size, dtype='f8')
    sqrt_w[ps_mask.flat != 0] = 0.

    # The linear least squares design and data matrices
    A = np.vstack([sqrt_w * psf_val, sqrt_w]).T
    b = sqrt_w * ps_exposure.flat

    # Extend the design and data matrices to incorporate priors
    A_priors = np.array([
        [1./stellar_flux_sigma, 0.], # Prior on stellar flux
        [0., 1./sky_sigma]           # Prior on sky level
    ])
    b_priors = np.array([
        stellar_flux_mean/stellar_flux_sigma, # Prior on stellar flux
        sky_mean/sky_sigma                    # Prior on sky level
    ])
    A = np.vstack([A, A_priors])
    b = np.hstack([b, b_priors])

    # Remove NaN and Inf values
    A[~np.isfinite(A)] = 0.
    b[~np.isfinite(b)] = 0.

    # Execute least squares
    star_flux, sky_level = np.linalg.lstsq(A, b)[0]

    return star_flux, sky_level


def fit_psf_coeffs(star_flux, star_sky,
                   star_x, star_y, ccd_shape,
                   ps_exposure, ps_weight, ps_mask,
                   sigma_nonzero_order=0.1):
    n_stars, n_x, n_y = ps_exposure.shape

    if not isinstance(sigma_nonzero_order, np.ndarray):
        sigma_nonzero_order = sigma_nonzero_order * np.ones((n_x, n_y), dtype='f8')

    # Scale coordinates so that x,y are each in range [-1,1]
    x = 2. * star_x / float(ccd_shape[0]) - 1.
    y = 2. * star_y / float(ccd_shape[1]) - 1.

    # Subtract each star's sky level from its postage stamp
    img_zeroed = ps_exposure - star_sky[:,None,None]
    img_zeroed[ps_mask != 0] = 0.

    # Transform pixel weights
    sqrt_w = np.sqrt(ps_weight)
    sqrt_w[ps_mask != 0] = 0. # Zero weight for masked pixels

    # Design matrix
    A_base = np.empty((n_stars+6, 6), dtype='f8') # without per-pixel weights
    A = np.empty((n_stars+6, 6), dtype='f8') # with weights - will be updated for each PSF pixel

    # TODO: Generalize this to arbitrary orders of x,y
    A_base[:n_stars,0] = 1.
    A_base[:n_stars,1] = eval_chebyt(1, x)
    A_base[:n_stars,2] = eval_chebyt(1, y)
    A_base[:n_stars,3] = eval_chebyt(2, x)
    A_base[:n_stars,4] = eval_chebyt(2, y)
    A_base[:n_stars,5] = eval_chebyt(1, x) * eval_chebyt(1, y)

    A_base[:n_stars,:] *= star_flux[:,None]

    # Data matrix
    b = np.zeros(n_stars+6, dtype='f8')

    # Remove masked pixels
    mask_pix = np.where(ps_mask != 0)[0]
    A_base[mask_pix] = 0.

    print '# of masked pixels: {}'.format(mask_pix.size)

    # Priors
    A_base[-6:,:] = np.diag(np.ones(6, dtype='f8'))

    psf_coeffs = np.empty((6, n_x, n_y), dtype='f8')

    # Loop over pixels in PSF, fitting coefficients for each pixel separately
    resid_tmp = np.empty((n_x, n_y), dtype='f8')

    for j in range(n_x):
        for k in range(n_y):
            # Design matrix
            A[:] = A_base[:]
            A[:n_stars,:] *= sqrt_w[:,None,j,k]
            A[-6] *= 0. # Prior on the zeroeth-order term
            A[-5:] *= 1 / sigma_nonzero_order[j,k] # Prior on higher-order terms

            # Data matrix
            b[:n_stars] = img_zeroed[:,j,k] * sqrt_w[:n_stars,j,k]
            b[mask_pix] = 0.

            # Remove NaN and Inf values
            A[~np.isfinite(A)] = 0.
            b[~np.isfinite(b)] = 0.

            # Execute least squares
            psf_coeffs[:,j,k], resid_tmp[j,k], _, _ = np.linalg.lstsq(A[:n_stars], b[:n_stars])#[0]
            #psf_coeffs[:,j,k], resid_tmp[j,k], _, _ = scipy.linalg.lstsq(A[:n_stars], b[:n_stars])#[0]

            '''
            if (j == 31) and (k == 31):
                bp = np.dot(A, psf_coeffs[:,j,k])
                resid = (b[:n_stars] - bp[:n_stars]) * sqrt_w[:,j,k]

                print ''
                print 'Ax - b:'
                print resid
                print ''

                pctiles = np.percentile(resid, [1., 50., 99.])
                width = 0.5 * (pctiles[-1] - pctiles[0])
                x0, x1 = pctiles[1] - width, pctiles[1] + width
                bins = np.linspace(x0, x1, 50)

                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(8,4), dpi=120)
                ax = fig.add_subplot(1,1,1)
                ax.hist(resid, bins, normed=1, histtype='stepfilled')
                plt.show()
            '''


    #np.set_printoptions(precision=3, linewidth=200)
    #print resid_tmp[29:35, 29:35] / 1000.
    #print resid_tmp[:5, :5] / 1000.

    return psf_coeffs


def normalize_psf_coeffs(psf_coeffs):
    '''
    Returns a copy of the PSF coefficients, in which the zeroeth-order PSF
    sums to unity. With the higher-order terms (which encode the variation
    across the CCD) added in, the PSF may not sum to unity.

    Input:
      psf_coeffs  The polynomial coefficients for each PSF pixel. The shape of
                  the output is (polynomial order, x, y).

    Output:
      psf_coeffs  A normalized copy of the input.
    '''

    norm = 1. / np.sum(psf_coeffs[0])

    return psf_coeffs * norm


def guess_psf(ps_exposure, ps_weight, ps_mask):
    '''
    Guess the PSF by stacking stars with no masked pixels.
    '''

    # Select postage stamps with no masked pixels
    idx = np.all(np.all(ps_mask == 0, axis=1), axis=1)
    tmp = ps_exposure[idx]

    # Normalize sum of each selected postage stamp to one
    tmp /= np.sum(np.sum(tmp, axis=1), axis=1)[:,None,None]

    # Take the median of the normalized postage stamps
    psf_guess = np.median(ps_exposure[idx], axis=0)

    # Normalize the guess to unity
    psf_guess /= np.sum(psf_guess)

    return psf_guess



def extract_stars(exposure_img, weight_img, mask_img, star_x, star_y,
                  width=int(np.ceil(5./0.263)), buffer_width=10, avoid_edges=1):
    '''
    Extracts postage stamps of stars from a CCD image.

    Input:
      exposure_img  CCD counts image.
      weight_img    CCD weight image.
      mask_img      CCD mask image.
      star_x        x-coordinate of each star on the CCD (in pixels - can be fractional).
      star_y        y-coordinate of each star on the CCD (in pixels - can be fractional).
      width         Width/height of the postage stamps.
      buffer_width  # of pixels to expand postage stamps by on each edge during
                    intermediate processing.
      avoid_edges   # of pixels on each edge of the CCD to avoid.

    Returns:
      ps_stack  An array of shape (3, n_stars, width, height). The zeroeth axis
                of the array corresponds to (exposure, weight, mask). The last
                two axes correspond to the width and height of each postage stamp.
    '''
    n_stars = star_x.size

    # Create empty stack of images
    w_ps = 2 * (width+buffer_width) + 1  # The width/height of the postage stamp, before final trimming
    ps_stack = np.zeros((3, n_stars, w_ps, w_ps), dtype='f8')
    ps_stack[2,:,:,:] = 1. # Initialize the mask to one (e.g., everything bad)

    # Determine amount to shift each star to center it on a pixel
    x_floor, y_floor = np.floor(star_x).astype('i4'), np.floor(star_y).astype('i4')
    dx = -(star_x - x_floor)
    dy = -(star_y - y_floor)

    # For each star, determine rectangle to copy from
    # the exposure (the "source" image), and the rectangle
    # to paste into in the postage stamp (the "destination" image)
    src_j0 = x_floor - (width+buffer_width)
    src_j1 = x_floor + (width+buffer_width) + 1
    src_k0 = y_floor - (width+buffer_width)
    src_k1 = y_floor + (width+buffer_width) + 1

    dst_j0 = np.zeros(n_stars, dtype='i4')
    dst_j1 = np.ones(n_stars, dtype='i4') * w_ps
    dst_k0 = np.zeros(n_stars, dtype='i4')
    dst_k1 = np.ones(n_stars, dtype='i4') * w_ps

    # Clip source rectangles at edges of exposure image, and shrink
    # destination rectangles accordingly
    idx = src_j0 < avoid_edges
    dst_j0[idx] = avoid_edges-src_j0[idx]
    src_j0[idx] = avoid_edges

    idx = src_j1 > exposure_img.shape[0] - avoid_edges
    dst_j1[idx] = exposure_img.shape[0] - avoid_edges - src_j1[idx]
    src_j1[idx] = exposure_img.shape[0] - avoid_edges

    idx = src_k0 < avoid_edges
    dst_k0[idx] = avoid_edges-src_k0[idx]
    src_k0[idx] = avoid_edges

    idx = src_k1 > exposure_img.shape[1] - avoid_edges
    dst_k1[idx] = exposure_img.shape[1] - avoid_edges - src_k1[idx]
    src_k1[idx] = exposure_img.shape[1] - avoid_edges

    kern = astropy.convolution.Box2DKernel(3)

    s = ps_stack.shape[2:]
    mask_center = np.ones(s, dtype=np.bool)
    mask_center[int(0.35*s[0]):int(0.65*s[0]), int(0.35*s[1]):int(0.65*s[1])] = 0

    sky_level = np.zeros(n_stars, dtype='f8')

    # Extract each star
    for i, (sj0,sj1,sk0,sk1,dj0,dj1,dk0,dk1) in enumerate(zip(src_j0,src_j1,
                                                              src_k0,src_k1,
                                                              dst_j0,dst_j1,
                                                              dst_k0,dst_k1)):
        #print '{}: ({},{},{},{}) --> ({},{},{},{})'.format(i,sj0,sj1,sk0,sk1,dj0,dj1,dk0,dk1)

        # Don't include postage stamps that are more than 50% clipped
        if (  (dj0 > 0.5*w_ps) or (dj1 < -0.5*w_ps)
           or (dk0 > 0.5*w_ps) or (dk1 < -0.5*w_ps)):
            continue

        # Extract star from exposure, weight and mask images
        tmp_exposure = exposure_img[sj0:sj1,sk0:sk1]
        tmp_weight = weight_img[sj0:sj1,sk0:sk1]
        tmp_mask = mask_img[sj0:sj1,sk0:sk1]

        # Skip star if no good pixels
        idx_use = (tmp_mask == 0)

        if np.all(~idx_use):
            continue

        # Copy star into postage stamp stack
        ps_stack[0,i,dj0:dj1,dk0:dk1] = tmp_exposure - np.median(tmp_exposure[idx_use])
        ps_stack[0,i] = sinc_shift_image(ps_stack[0,i], dx[i], dy[i])

        ps_stack[1,i] = np.median(tmp_weight[idx_use])
        ps_stack[1,i,dj0:dj1,dk0:dk1] = tmp_weight
        #ps_stack[1,i] = 1. / sinc_shift_image(1./ps_stack[1,i], dx[i], dy[i])   # Shift 1/weight, b/c better behaved
        ps_stack[1,i] = sinc_shift_image(ps_stack[1,i], dx[i], dy[i])

        ps_stack[2,i,dj0:dj1,dk0:dk1] = tmp_mask
        ps_stack[2,i] = astropy.convolution.convolve(ps_stack[2,i], kern, boundary='extend')

        # Subtract an estimate of the sky level, using non-central pixels
        idx_use = (ps_stack[2,i] == 0) & mask_center
        sky_level[i] = np.median(ps_stack[0,i][idx_use])
        ps_stack[0,i] -= sky_level[i]

    # Don't allow negative weights
    idx = (ps_stack[1] < 0.)
    ps_stack[1,idx] = 0.

    #print 'sky levels:'
    #print sky_level
    #print ''

    # Clip edge pixels off of postage stamps and return result
    return ps_stack[:, :, buffer_width:w_ps-buffer_width, buffer_width:w_ps-buffer_width]


def filter_postage_stamps(ps_mask, min_pixel_fraction=0.5):
    '''
    Return the indices of the stellar postage stamps that have enough good
    pixels to use.

    Inputs:
      ps_mask             Postage stamp of the mask in the vicinity of the star.
      min_pixel_fraction  Minimum fraction of good pixels to accept a stellar
                          postage stamp.

    Output:
      keep_idx  The indices of the postge stamps with enough good pixels.
    '''

    n_pix = ps_mask.shape[1] * ps_mask.shape[2]
    n_good_pix = np.sum(np.sum(ps_mask == 0, axis=1), axis=1)
    idx = (n_good_pix > min_pixel_fraction * n_pix)

    return idx


def get_star_locations(ps1_table, wcs, ccd_shape, min_separation=10./0.263):
    '''
    Get pixel coordinates of PS1 stars that fall on a given DECam CCD exposure.

    Inputs:
      ps1_table  A table of PS1 detections that fall on the given exposure.
      wcs        A World Coordinate System object describing the projection
                 of the CCD.
      ccd_shape  The shape (in pixels) of the CCD: (x extent, y extent).

    Optional parameters:
      min_separation  Minimum separation (in pixels) between stars. Stars closer
                      to one another than this distance will be rejected.

    Outputs:
      star_x  x-coordinates (in pixels) of the selected PS1 stars on the CCD.
      star_y  y-coordinates (in pixels) of the selected PS1 stars on the CCD.
      star_ps1_mag  PS1 grizy magnitudes of stars.
    '''

    # Get stellar pixel coordinates
    star_y, star_x = wcs.wcs_world2pix(ps1_table['RA'], ps1_table['DEC'], 0)   # 0 is the coordinate in the top left (the numpy, but not FITS standard)

    # Filter stars that are off the CCD
    idx = ((star_x > -min_separation) & (star_x < ccd_shape[0] + min_separation) &
           (star_y > -min_separation) & (star_y < ccd_shape[1] + min_separation))

    star_x = star_x[idx]
    star_y = star_y[idx]
    ps1_table = ps1_table[idx]

    # Filter stars that are too close to one another
    idx = filter_close_points(star_x, star_y, r=min_separation)

    # Filter stars that don't pass quality cuts
    idx &= filter_ps1_quality(ps1_table)

    return star_x[idx], star_y[idx], ps1_table['MEAN'][idx]


def filter_ps1_quality(ps1_table):
    '''
    Return the indices of PS1 objects that pass a set of cuts, which select for
    compact sources detected in multiple exposures.

    Input:
      ps1_table: A record array of PS1 objects, containing at least:
                   - 'NMAG_OK' (number of good detections in each band)
                   - 'MEAN'    (mean psf mag in each band over multiple detections)
                   - 'MEAN_AP' (mean aperture mag in each band over multiple detections)

    Output:
      keep_idx  A boolean array, containing True for stars that pass the cuts,
                and False otherwise.
    '''

    idx = ((np.sum(ps1_table['NMAG_OK'], axis=1) >= 5) &
           (np.sum(ps1_table['MEAN'] - ps1_table['MEAN_AP'] < 0.1, axis=1) >= 2))

    return idx


def get_ps1_stars_for_ccd(wcs, ccd_shape, min_separation):
    '''
    Returns pixel coordinates for stars detected by PS1 that fall on the CCD.

    Inputs:
      wcs             A World Coordinate System object describing the projection
                      of the CCD.
      ccd_shape       The shape (in pixels) of the CCD: (x extent, y extent).
      min_separation  Minimum separation (in pixels) between stars. Stars closer
                      to one another than this distance will be rejected.

    Ouptuts:
      star_x  x-coordinates of the PS1 stars (in pixels).
      star_y  y-coordinates of the PS1 stars (in pixels).
      star_ps1_mag  PS1 grizy magnitudes of stars.
    '''

    # Load locations of PS1 stars
    fname = 'psftest/ps1stars-c4d_150109_051822.fits'
    ps1_table = astropy.io.fits.getdata(fname, 1)
    # TODO: Replace this with call to ps1cat.ps1cat

    star_x, star_y, star_ps1_mag = get_star_locations(ps1_table, wcs, ccd_shape,
                                                      min_separation=min_separation)

    return star_x, star_y, star_ps1_mag


def calc_star_chisq(psf_coeffs, ps_exposure, ps_weight, ps_mask,
                    star_x, star_y, star_flux, sky_level, ccd_shape,
                    return_chisq_img=False):
    '''
    Calculate the mean squared deviation of each postage stamp's pixels from the
    modeled flux (based on the PSF model, fitted stellar flux and sky level),
    weighted by the exposure weights.

    Inputs:
      psf_coeffs
      ps_exposure
      ps_weight
      ps_mask
      star_x
      star_y
      star_flux
      sky_level
      ccd_shape

    Optional parameters:
      return_chisq_img  If true, a stack of images showing chi^2 per pixel is
                        returned. Default: False.

    Outputs:
      psf_resid  Mean squared weighted residuals between the postage stamps and
                 the modeled flux.
    '''

    # Generate an image of the PSF at the location of each star
    x = 2. * star_x / float(ccd_shape[0]) - 1.
    y = 2. * star_y / float(ccd_shape[1]) - 1.

    psf_img = np.zeros(ps_exposure.shape, dtype='f8')   # Shape: (star, x, y)
    psf_img += psf_coeffs[0,None,:,:]
    psf_img += psf_coeffs[1,None,:,:] * eval_chebyt(1, x)[:,None,None]
    psf_img += psf_coeffs[2,None,:,:] * eval_chebyt(1, y)[:,None,None]
    psf_img += psf_coeffs[3,None,:,:] * eval_chebyt(2, x)[:,None,None]
    psf_img += psf_coeffs[4,None,:,:] * eval_chebyt(2, y)[:,None,None]
    psf_img += psf_coeffs[5,None,:,:] * (eval_chebyt(1, x)*eval_chebyt(1, y))[:,None,None]

    # Normalize each PSF to unity
    psf_norm = 1. / np.sum(np.sum(psf_img, axis=1), axis=1)
    psf_img *= psf_norm[:,None,None]

    # Transform the PSF into a model of counts
    psf_img *= star_flux[:,None,None]
    psf_img += sky_level[:,None,None]

    # Calculate an image of chi^2 of each star
    chisq_img = ps_exposure - psf_img
    chisq_img *= chisq_img * ps_weight

    idx_mask = (ps_mask >= 1.e-5) | ~np.isfinite(chisq_img)
    chisq_img[idx_mask] = 0.

    # Calculate chi^2/dof for each star
    chisq = np.sum(np.sum(chisq_img, axis=1), axis=1)
    n_pix = np.sum(np.sum((ps_mask == 0).astype('f8'), axis=1), axis=1)
    chisq /= n_pix #float(ps_exposure.shape[1] * ps_exposure.shape[2])

    print ''
    print 'n_pix:'
    print n_pix
    print ''

    if return_chisq_img:
        chisq_img[idx_mask] = np.nan
        return chisq, chisq_img

    return chisq


def extract_psf(exposure_img, weight_img, mask_img, wcs,
                min_separation=50., min_pixel_fraction=0.5, n_iter=1,
                psf_halfwidth=31, buffer_width=10, avoid_edges=1,
                star_chisq_threshold=2., max_star_shift=3.,
                pixel_chisq_threshold=10.**2., sky_sigma=0.05,
                sigma_nonzero_order=0.5, return_postage_stamps=False):
    '''
    Extract the PSF from a CCD exposure, using a pixel basis. Each pixel is
    represented as a polynomial in (x,y), where x and y are the pixel
    coordinates on the CCD.

    Inputs:
      exposure_img  CCD counts image.
      weight_img    CCD weight image.
      mask_img      CCD mask image.
      wcs           A World Coordinate System object describing the projection
                    of the CCD.

    Optional parameters:
      n_iter              # of iterations to run for. Each iteration consists of
                          fitting the stellar fluxes and local sky levels, given
                          the current PSF fit, and then updating the PSF fit,
                          based on the stars.
      psf_halfwidth       The width/height of the PSF image will be 2*psf_halfwidth+1.
      min_separation      Minimum separation (in pixels) between stars. Stars closer
                          to one another than this distance will be rejected.
      min_pixel_fraction  Minimum fraction of pixels around a star that must be
                          unmasked in order for the star to be used for the fit.
      buffer_width        # of pixels to expand postage stamps by on each edge
                          during intermediate processing.
      avoid_edges         # of pixels on each edge of the CCD to avoid.
      star_chisq_threshold   chi^2/dof at which stars will not be used to derive
                             PSF fit.
      max_star_shift         The maximum shift (in pixels) that can be applied
                             to the position of any star (based on an estimate
                             of the star's centroid location).
      return_postage_stamps  If True, return a stack of postage stamps of the
                             stars used in the fit.

    Output:
      psf_coeffs  The polynomial coefficients for each PSF pixel. The shape of
                  the output is (polynomial order, x, y).

    If return_postage_stamps == True, a dictionary containing the following
    keys is also returned:
      ps_exposure   Counts postage stamps of the stars used in the fit. The
                    shape is (# of stars, x, y).
      ps_weight     Weight postage stamps for the stars used in the fit. The
                    shape is (# of stars, x, y).
      ps_mask       Mask postage stamps for the stars used in the fit. The shape
                    is (# of stars, x, y).
      star_x        x-coordinates (in pixels) on the CCD of the stars used in
                    the fit.
      star_y        y-coordinates (in pixels) on the CCD of the stars used in
                    the fit.
      star_PS1_mag  PS1 grizy magnitudes of the stars.
      stellar_flux  Flux of each star (as a multiple of the local PSF) used in
                    the fit.
      sky_level     The local sky level for each star used in the fit.
    '''

    # Select stars (from PS1)
    ccd_shape = exposure_img.shape
    star_x, star_y, star_ps1_mag = get_ps1_stars_for_ccd(wcs, ccd_shape,
                                                         min_separation=min_separation)

    # Extract centered postage stamps of stars, sinc shifting stars as necessary
    ps_exposure, ps_weight, ps_mask = extract_stars(exposure_img, weight_img,
                                                    mask_img, star_x, star_y,
                                                    width=psf_halfwidth,
                                                    buffer_width=buffer_width,
                                                    avoid_edges=avoid_edges)

    # Recenter the postage stamps
    ps_exposure, ps_weight, ps_mask = recenter_postage_stamps(
        exposure_img, weight_img, mask_img,
        star_x, star_y,
        ps_exposure, ps_weight, ps_mask,
        max_shift=max_star_shift,
        width=psf_halfwidth,
        buffer_width=buffer_width,
        avoid_edges=avoid_edges
    )

    # Filter out stars that are more than a certain percent masked
    idx = filter_postage_stamps(ps_mask, min_pixel_fraction=min_pixel_fraction)
    ps_exposure = ps_exposure[idx]
    ps_weight = ps_weight[idx]
    ps_mask = ps_mask[idx]
    star_x = star_x[idx]
    star_y = star_y[idx]
    star_ps1_mag = star_ps1_mag[idx]

    # Find indices of stars with no masking at all
    idx_unmasked = filter_postage_stamps(ps_mask, min_pixel_fraction=0.99999)
    print '{} completely unmasked stars.'.format(np.sum(idx_unmasked))

    # Guess the PSF by median-stacking pristine postage stamps (no masked pixels)
    psf_guess = guess_psf(ps_exposure, ps_weight, ps_mask)

    # The fit parameters
    n_stars = ps_exposure.shape[0]
    psf_coeffs = np.zeros((6, ps_exposure.shape[1], ps_exposure.shape[2]), dtype='f8')
    psf_coeffs[0,:,:] = psf_guess[:,:]
    stellar_flux = np.empty(n_stars, dtype='f8')            # The true flux
    stellar_flux_improper = np.empty(n_stars, dtype='f8')   # Flux defined as a multiple of the PSF (which may be unnormalized)
    sky_level = np.empty(n_stars, dtype='f8')
    stellar_flux[:] = np.nan
    sky_level[:] = np.nan

    # Fit a 2D Gaussian to the PSF guess, to be used as a prior on the
    # zeroeth-order PSF coefficients.
    #gauss_psf_prior = fit_2d_gaussian(psf_coeffs[0,:,:])
    #print gauss_psf_prior

    stellar_flux_mean = np.zeros(n_stars, dtype='f8')
    stellar_flux_sigma = np.empty(n_stars, dtype='f8')
    stellar_flux_sigma[:] = np.inf

    star_dx = np.zeros(n_stars, dtype='f8')
    star_dy = np.zeros(n_stars, dtype='f8')

    ps_mask_effective = None
    ps_chisq_mask = None

    #star_chisq_threshold_iter = np.linspace(10. * star_chisq_threshold, star_chisq_threshold, n_iter)

    # Iterate linear fits to reach the solution
    for j in range(n_iter):
        # Fit the flux and local sky level for each star
        for k in range(n_stars):
            stellar_flux[k], sky_level[k] = fit_star_params(
                psf_coeffs,
                star_x[k], star_y[k],
                ps_exposure[k], ps_weight[k],
                ps_mask[k], ccd_shape,
                sky_sigma=sky_sigma,
                stellar_flux_mean=stellar_flux_mean[k],
                stellar_flux_sigma=stellar_flux_sigma[k]
            )

        # Calculate the "improper" fluxes (defined in unnormalized PSF units)
        psf_flux = calc_psf_fluxes(psf_coeffs, star_x, star_y, ccd_shape)
        stellar_flux_improper = stellar_flux / psf_flux

        # Flag stars with bad chi^2/dof
        star_chisq, star_chisq_img = calc_star_chisq(
            psf_coeffs,
            ps_exposure, ps_weight, ps_mask,
            star_x, star_y,
            stellar_flux, sky_level,
            ccd_shape,
            return_chisq_img=True
        )
        idx_chisq = (star_chisq < star_chisq_threshold)

        print ''
        print 'star chi^2:'
        print star_chisq
        print ''

        # Flag pixels with bad chi^2/dof
        ps_chisq_mask = (np.abs(star_chisq_img) > star_chisq_threshold).astype('f8')
        #print '{:.2f} % of pixels masked'.format(np.sum(ps_chisq_mask) / float(ps_chisq_mask.size))
        kern = astropy.convolution.Box2DKernel(3)
        #for k in range(n_stars):
        #    ps_chisq_mask[k] = astropy.convolution.convolve(ps_chisq_mask[k], kern,
        #                                                    boundary='extend')
        ps_chisq_mask = (ps_chisq_mask < 0.1)
        print '{:.2f} % of pixels masked'.format(np.sum(ps_chisq_mask) / float(ps_chisq_mask.size))

        print 'Rejected stars:'
        print np.where(~idx_chisq)[0]

        print 'Rejected {} of {} stars.'.format(np.sum(~idx_chisq), idx_chisq.size)

        # Update the prior on stellar flux
        idx = idx_chisq & idx_unmasked
        flux_predictor = gen_stellar_flux_predictor(
            stellar_flux[idx], star_ps1_mag[idx],
            star_x[idx], star_y[idx],
            ccd_shape, psf_coeffs
        )
        # TODO: Make flux predictor robust against missing PS1 bands and uncomment following lines
        #stellar_flux_mean[:] = flux_predictor(star_ps1_mag)
        #stellar_flux_sigma[:] = np.sqrt(stellar_flux_mean)

        # Prior on nonzero-order PSF coefficients
        sigma_tmp = 0.01 * psf_coeffs[0]
        if j > 5:
            sigma_tmp = sigma_nonzero_order * psf_coeffs[0]

        sigma_tmp[sigma_tmp < 0.] = 0.
        sigma_tmp = np.sqrt(sigma_tmp**2. + 0.01**2.)

        # Fit the PSF coefficients in each pixel
        if j == 0:
            idx = idx_chisq & idx_unmasked
        else:
            idx = idx_chisq

        if j > 100:
            ps_mask_effective = ((ps_mask[idx] != 0) | (ps_chisq_mask[idx] > 0.1)).astype('f8')
        else:
            ps_mask_effective = ps_mask[idx]

        psf_coeffs = fit_psf_coeffs(stellar_flux_improper[idx], sky_level[idx],
                                    star_x[idx], star_y[idx],
                                    ccd_shape, ps_exposure[idx],
                                    ps_weight[idx], ps_mask_effective,
                                    sigma_nonzero_order=sigma_tmp)

        # Normalize the PSF
        psf_coeffs = normalize_psf_coeffs(psf_coeffs)

        if j > 1:
            # Recenter stars
            for k in range(n_stars):
                star_dx[k], star_dy[k] = fit_star_offset(
                    psf_coeffs,
                    ps_exposure[k],
                    ps_weight[k],
                    ps_mask[k],
                    star_x[k],
                    star_y[k],
                    stellar_flux[k],
                    sky_level[k],
                    ccd_shape,
                    max_shift=2.
                ) # TODO: include max_shift as option

            star_x += star_dx
            star_y += star_dy

            # Re-extract stars, with new shifts
            ps_exposure, ps_weight, ps_mask = extract_stars(
                exposure_img, weight_img,
                mask_img, star_x, star_y,
                width=psf_halfwidth,
                buffer_width=buffer_width,
                avoid_edges=avoid_edges
            )


    if return_postage_stamps:
        for k in range(n_stars):
            stellar_flux[k], sky_level[k] = fit_star_params(
                psf_coeffs,
                star_x[k], star_y[k],
                ps_exposure[k], ps_weight[k],
                ps_mask[k], ccd_shape,
                sky_sigma=sky_sigma,
                stellar_flux_mean=stellar_flux_mean[k],
                stellar_flux_sigma=stellar_flux_sigma[k]
            )

        star_dict = {
            'ps_exposure':   ps_exposure,
            'ps_weight':     ps_weight,
            'ps_mask':       ps_mask,
            'ps_chisq_mask': ps_chisq_mask,
            'star_x':        star_x,
            'star_y':        star_y,
            'star_ps1_mag':  star_ps1_mag,
            'stellar_flux':  stellar_flux,
            'sky_level':     sky_level
        }

        return psf_coeffs, star_dict

    return psf_coeffs


def load_exposure(fname_pattern, ccd_id):
    '''
    Load one CCD from an exposure.

    Inputs:
      fname_pattern  A filename of the form c4d_150109_051822_oo{}_z_v1.fits.fz,
                     where {} will be expanded in order to select the image,
                     weight and mask files.
      ccd_id         The identifier of the desired CCD (e.g., 'S31').

    Outputs:
      img_data     Exposure image
      weight_data  Weight image
      mask_data    Mask image
      wcs          The WCS astrometric solution
      exp_num      The exposure number
    '''

    hdulist = pyfits.open(fname_pattern.format('i'))
    exp_num = hdulist[0].header['EXPNUM']

    #primary_data, primary_header = pyfits.getdata(fname_pattern.format('i'), 0,
    #                                              header=True)
    #print primary_header.keys()
    #exp_num = primary_header['EXPNUM']
    img_data = hdulist[ccd_id].data[:]
    img_header = hdulist[ccd_id].header

    #img_data, img_header = pyfits.getdata(fname_pattern.format('i'),
    #                                      ccd_id, header=True)
    wcs = astropy.wcs.WCS(header=img_header)

    weight_data = pyfits.getdata(fname_pattern.format('w'), ccd_id)
    mask_data = pyfits.getdata(fname_pattern.format('d'), ccd_id)
    #mask_data[:] = 0.

    # Fix the weights
    weight_data[weight_data < 0.] = 0.

    # Apply the mask to the weights (and zero out the corresponding image pixels)
    #mask_idx = (mask_data != 0)
    #img_data[mask_idx] = 0.
    #weight_data[mask_idx] = 0.

    return img_data, weight_data, mask_data, wcs, exp_num


def write_psf_file(fname, psf_coeffs, ccd_shape, ccd_name, exp_num):
    '''
    Write the PSF coefficients to a FITS file.
    '''

    primary = pyfits.PrimaryHDU()
    primary.header['PSFTYPE'] = 'PCA-PIX'
    primary.header['YSCALE'] = 1. / float(ccd_shape[0])
    primary.header['XSCALE'] = 1. / float(ccd_shape[1])
    primary.header['NPIX_Y'] = ccd_shape[0]
    primary.header['NPIX_X'] = ccd_shape[1]
    primary.header['EXPNUM'] = exp_num

    hdu1 = pyfits.TableHDU.from_columns([
        pyfits.Column(name='IMODEL', format='J', array=range(6)),
        pyfits.Column(name='YEXP', format='J', array=[0,1,0,2,0,1]),
        pyfits.Column(name='XEXP', format='J', array=[0,0,1,0,2,1])
    ])

    hdu2 = pyfits.ImageHDU(data=psf_coeffs, name=ccd_name)
    hdu2.header['CCD'] = ccd_name

    hdulist = pyfits.HDUList([primary, hdu1, hdu2])
    hdulist.writeto(fname, clobber=True, checksum=True)
