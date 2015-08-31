#!/usr/env python

import numpy as np
import scipy.ndimage

import astropy.wcs
import astropy.io
import astropy.convolution

from sklearn.neighbors import NearestNeighbors


def img_center_of_mass(img, img_mask):
    idx = ~np.isfinite(img) | (img_mask != 0) # Determine image mask
    idx |= idx[:,::-1] | idx[::-1,:] # Make mask symmetric under parity flips
    img_clean = np.empty(img.shape, dtype='f8')
    img_clean[:] = img[:]
    img_clean[idx] = 0.

    print np.sum(~np.isfinite(img_clean))
    print np.sum(img_clean)

    return scipy.ndimage.measurements.center_of_mass(img_clean)


def gen_centered_weights(shape, sigma):
    '''
    Return a weight image that weights pixels towards the center of the image
    more strongly than pixels away from the center.
    '''
    return np.array(astropy.convolution.Gaussian2DKernel(
        sigma, x_size=shape[0], y_size=shape[1],
        mode='oversample', factor=5
    ))


def recenter_postage_stamps(exposure_img, weight_img, mask_img, star_x, star_y,
                            ps_exposure, ps_weight, ps_mask, **kwargs):
    '''
    Recenter a stack of postage stamps, trying to properly centroid each star.
    Should work for small shifts (< 1 pixel). Calculates a weighted center of
    mass for each postage stamp (the weighting ignores pixels away from the
    center of the image), and then going back into the original exposure image
    and extracting the stars again.
    '''

    max_shift = kwargs.pop('max_shift', 2.)

    # Weight to apply to each image before calculating center of mass
    w_cent = gen_centered_weights(ps_exposure.shape[1:], 3.)

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

    print 'x shifts:'
    print star_x_cent - np.array(star_x)
    print ''
    print 'y shifts:'
    print star_y_cent - np.array(star_y)
    print ''
    print 'dx percentiles:'
    print np.percentile(dxy[idx_good,0], [1., 10., 50., 90., 99.])
    print 'dy percentiles:'
    print np.percentile(dxy[idx_good,1], [1., 10., 50., 90., 99.])
    print ''

    ret = extract_stars(exposure_img, weight_img, mask_img,
                        star_x_cent, star_y_cent, **kwargs)

    return ret


def sinc_shift_image(img, dx, dy):
    '''
    Shift a real-valued image by (dx,dy), using the DFT shift theorem.

    Input:
      img  A 2D image, represented by a 2D, real-valued numpy array.
      dx   The shift along the x-axis, in pixels
      dy   The shift along the y-axis, in pixels

    Output:
      A shifted copy of the image, as a 2D numpy array.
    '''

    img_dft = np.fft.fft2(img)
    img_dft = scipy.ndimage.fourier.fourier_shift(img_dft, (dx,dy))
    return np.real(np.fft.ifft2(img_dft))


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
    x = star_x / float(ccd_shape[0])
    y = star_y / float(ccd_shape[1])

    psf_img = np.empty((psf_coeffs.shape[1], psf_coeffs.shape[2]), dtype='f8')

    # TODO: Generalize this to arbitrary orders of x,y
    psf_img[:,:] = psf_coeffs[0,:,:]
    psf_img[:,:] += psf_coeffs[1,:,:] * x
    psf_img[:,:] += psf_coeffs[2,:,:] * y
    psf_img[:,:] += psf_coeffs[3,:,:] * x*x
    psf_img[:,:] += psf_coeffs[4,:,:] * y*y
    psf_img[:,:] += psf_coeffs[5,:,:] * x*y

    return psf_img




def fit_star_params(psf_coeffs, star_x, star_y,
                    ps_exposure, ps_weight, ps_mask, ccd_shape,
                    sky_mean=0., sky_sigma=0.5):
    '''
    Fit stellar flux and sky brightness, given a PSF model.
    '''

    # Evaluate the PSF at the location of the star
    psf_val = eval_psf(psf_coeffs, star_x, star_y, ccd_shape)
    #psf_val /= np.sum(psf_val)
    psf_val.shape = (psf_val.size,)

    # Calculate the square root of the weight
    sqrt_w = np.sqrt(ps_weight.flat)
    sqrt_w[ps_mask.flat != 0] = 0.

    # The linear least squares design and data matrices
    A = np.vstack([sqrt_w * psf_val, sqrt_w]).T
    b = sqrt_w * ps_exposure.flat

    # Extend the design and data matrices to incorporate priors
    A_priors = np.array([
        [0., 1./sky_sigma]  # Prior on sky level
    ])
    b_priors = np.array([
        sky_mean/sky_sigma  # Prior on sky level
    ])
    A = np.vstack([A, A_priors])
    b = np.hstack([b, b_priors])

    # Remove NaN and Inf values
    A[~np.isfinite(A)] = 0.
    b[~np.isfinite(b)] = 0.

    # Execute least squares
    a0, a1 = np.linalg.lstsq(A, b)[0]

    return a0, a1


def fit_psf_coeffs(star_flux, star_sky,
                   star_x, star_y, ccd_shape,
                   ps_exposure, ps_weight, ps_mask,
                   sigma_nonzero_order=0.1):
    n_stars, n_x, n_y = ps_exposure.shape

    # Scale coordinates so that x,y are each in range [0,1]
    x = star_x / float(ccd_shape[0])
    y = star_y / float(ccd_shape[1])

    # Normalize counts (by removing sky background and dividing out stellar flux)
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
    A_base[:n_stars,1] = x
    A_base[:n_stars,2] = y
    A_base[:n_stars,3] = x**2.
    A_base[:n_stars,4] = y**2.
    A_base[:n_stars,5] = x*y

    # Data matrix
    b = np.zeros(n_stars+6, dtype='f8')

    # Priors
    A_base[-6:,:] = np.diag(np.ones(6, dtype='f8'))

    psf_coeffs = np.empty((6, n_x, n_y), dtype='f8')

    # Loop over pixels in PSF, fitting coefficients for each pixel separately
    for j in range(n_x):
        for k in range(n_y):
            # Design matrix
            A[:] = A_base[:]
            #A[:n_stars,:] *= sqrt_w[:,None,j,k]
            A[-6] *= 0.
            A[-5:] *= 1. / np.sqrt(sigma_nonzero_order)

            #print ''
            #print A[-6:]
            #print ''

            # Data matrix
            b[:n_stars] = img_zeroed[:,j,k] * star_flux[:] #* sqrt_w[:n_stars,j,k]

            # Remove NaN and Inf values
            A[~np.isfinite(A)] = 0.
            b[~np.isfinite(b)] = 0.

            # Execute least squares
            psf_coeffs[:,j,k] = np.linalg.lstsq(A, b)[0]

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
    dx = -(star_x - x_floor - 0.5)
    dy = -(star_y - y_floor - 0.5)

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
        ps_stack[1,i] = sinc_shift_image(ps_stack[1,i], dx[i], dy[i])

        ps_stack[2,i,dj0:dj1,dk0:dk1] = tmp_mask
        ps_stack[2,i] = astropy.convolution.convolve(ps_stack[2,i], kern, boundary='extend')#white_tophat(ps_stack[2,i], size=3, mode='nearest')

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
                    star_x, star_y, star_flux, sky_level, ccd_shape):
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

    Outputs:
      psf_resid  Mean squared weighted residuals between the postage stamps and
                 the modeled flux.
    '''

    x = star_x / float(ccd_shape[0])
    y = star_y / float(ccd_shape[1])

    psf_img = np.zeros(ps_exposure.shape, dtype='f8')
    psf_img += psf_coeffs[0,None,:,:]
    psf_img += psf_coeffs[1,None,:,:] * x[:,None,None]
    psf_img += psf_coeffs[2,None,:,:] * y[:,None,None]
    psf_img += psf_coeffs[3,None,:,:] * x[:,None,None] * x[:,None,None]
    psf_img += psf_coeffs[4,None,:,:] * y[:,None,None] * y[:,None,None]
    psf_img += psf_coeffs[5,None,:,:] * x[:,None,None] * y[:,None,None]

    psf_img *= star_flux[:,None,None]
    psf_img += sky_level[:,None,None]

    psf_resid = psf_img - ps_exposure
    psf_resid *= psf_resid * ps_weight
    psf_resid[(ps_mask != 0) | ~np.isfinite(psf_resid)] = 0.
    psf_resid = np.sum(np.sum(psf_resid, axis=1), axis=1)
    psf_resid /= float(ps_exposure.shape[1] * ps_exposure.shape[2])

    return psf_resid


def extract_psf(exposure_img, weight_img, mask_img, wcs,
                min_separation=50., min_pixel_fraction=0.5, n_iter=1,
                psf_halfwidth=31, buffer_width=10, avoid_edges=1,
                star_chisq_threshold=2., return_postage_stamps=False):
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

    # Guess the PSF by median-stacking pristine postage stamps (no masked pixels)
    psf_guess = guess_psf(ps_exposure, ps_weight, ps_mask)

    # The fit parameters
    n_stars = ps_exposure.shape[0]
    psf_coeffs = np.zeros((6, ps_exposure.shape[1], ps_exposure.shape[2]), dtype='f8')
    psf_coeffs[0,:,:] = psf_guess[:,:]
    stellar_flux = np.empty(n_stars, dtype='f8')
    sky_level = np.empty(n_stars, dtype='f8')

    for j in range(n_iter):
        # Fit the flux and local sky level for each star
        for k in range(n_stars):
            stellar_flux[k], sky_level[k] = fit_star_params(psf_coeffs,
                                                            star_x[k], star_y[k],
                                                            ps_exposure[k], ps_weight[k],
                                                            ps_mask[k], ccd_shape)

        # Flag stars with bad chi^2/dof
        star_chisq = calc_star_chisq(psf_coeffs, ps_exposure, ps_weight, ps_mask,
                                     star_x, star_y, stellar_flux, sky_level,
                                     ccd_shape)
        idx = (star_chisq < star_chisq_threshold)

        print 'Rejected stars:'
        print np.where(~idx)[0]

        print 'Rejected {} of {} stars.'.format(np.sum(~idx), idx.size)

        # Fit the PSF coefficients in each pixel
        psf_coeffs = fit_psf_coeffs(stellar_flux[idx], sky_level[idx],
                                    star_x[idx], star_y[idx],
                                    ccd_shape, ps_exposure[idx],
                                    ps_weight[idx], ps_mask[idx])

        # Normalize the PSF
        psf_coeffs = normalize_psf_coeffs(psf_coeffs)

    if return_postage_stamps:
        for k in range(n_stars):
            stellar_flux[k], sky_level[k] = fit_star_params(psf_coeffs,
                                                            star_x[k], star_y[k],
                                                            ps_exposure[k], ps_weight[k],
                                                            ps_mask[k], ccd_shape)

        star_dict = {
            'ps_exposure':  ps_exposure,
            'ps_weight':    ps_weight,
            'ps_mask':      ps_mask,
            'star_x':       star_x,
            'star_y':       star_y,
            'star_ps1_mag': star_ps1_mag,
            'stellar_flux': stellar_flux,
            'sky_level':    sky_level
        }

        return psf_coeffs, star_dict

    return psf_coeffs


def test_sinc_shift_image():
    from PIL import Image
    im = Image.open('psftest/shift.png')
    img = np.sum(np.array(im)[:,:,:3], axis=2)

    import matplotlib.pyplot as plt
    fig = plt.figure()

    for j in range(3):
        for k in range(3):
            ax = fig.add_subplot(3,3,3*k+j+1)
            img_shifted = np.real(sinc_shift_image(img, 50*j, 50*k))
            vmin, vmax = np.percentile(img, [1., 99.])
            ax.imshow(img_shifted, origin='upper', aspect='equal',
                           interpolation='none', cmap='binary',
                           vmin=vmin, vmax=vmax)

    plt.show()


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
    '''

    img_data, img_header = astropy.io.fits.getdata(fname_pattern.format('i'),
                                                   ccd_id, header=True)
    wcs = astropy.wcs.WCS(header=img_header)

    weight_data = astropy.io.fits.getdata(fname_pattern.format('w'), ccd_id)
    mask_data = astropy.io.fits.getdata(fname_pattern.format('d'), ccd_id)

    # Apply the mask to the weights (and zero out the corresponding image pixels)
    #mask_idx = (mask_data != 0)
    #img_data[mask_idx] = 0.
    #weight_data[mask_idx] = 0.

    return img_data, weight_data, mask_data, wcs


def test_find_star_centers():
    pass


def test_load_exposure():
    # Load a test image
    fname_pattern = 'psftest/c4d_150109_051822_oo{}_z_v1.fits.fz'
    img_data, weight_data, mask_data, wcs = load_exposure(fname_pattern, 'S31')

    # Load locations of PS1 stars
    fname = 'psftest/ps1stars-c4d_150109_051822.fits'
    ps1_table = astropy.io.fits.getdata(fname, 1)
    star_x, star_y = get_star_locations(ps1_table, wcs, img_data.shape, min_separation=50)

    # Extract postage stamps of stars
    ps_exposure, ps_weight, ps_mask = extract_stars(img_data, weight_data, mask_data,
                                                    star_x, star_y)

    # Plot the CCD image and weight, with stellar locations from PS1 overplotted
    vmin, vmax = np.percentile(img_data[(img_data > 1.) & (mask_data == 0)], [1.,99.])

    import matplotlib.pyplot as plt
    fig = plt.figure(dpi=300)

    ax = fig.add_subplot(2,1,1)
    ax.imshow(img_data.T, origin='upper', aspect='equal', interpolation='nearest',
                          cmap='binary_r', vmin=vmin, vmax=vmax)

    ax.scatter(star_x, star_y,
               s=12, edgecolor='b', facecolor='none',
               lw=0.75, alpha=0.75)

    ax.set_xlim(0, img_data.shape[0])
    ax.set_ylim(0, img_data.shape[1])

    ax = fig.add_subplot(2,1,2)
    sigma = 1./np.sqrt(weight_data)
    vmin_s, vmax_s = np.percentile(sigma[np.isfinite(sigma) & (mask_data == 0)], [1., 99.])
    ax.imshow(sigma.T, origin='upper', aspect='equal', interpolation='nearest',
                       cmap='binary_r', vmin=vmin_s, vmax=vmax_s)

    fig.savefig('ccd_with_ps1_detections_v2.png', dpi=300, bbox_inches='tight')

    # Plot the postage stamps
    n_ps = ps_exposure.shape[0]

    n_x = int(np.ceil(np.sqrt(n_ps)))
    n_y = int(np.ceil(n_ps/float(n_x)))

    fig = plt.figure(figsize=(n_x,n_y), dpi=100)

    for k in range(n_ps):
        tmp = ps_exposure[k]
        tmp[ps_mask[k] != 0] = np.nan

        vmax = 1.

        idx = (tmp > 1.)
        if np.any(idx):
            vmax = np.percentile(np.abs(tmp[idx]), 99.5)

        ax = fig.add_subplot(n_x, n_y, k+1, axisbg='g')

        ax.imshow(tmp.T, origin='upper', aspect='equal',
                  interpolation='nearest', cmap='bwr_r',
                  vmin=-vmax, vmax=vmax)

        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    fig.savefig('postage_stamps.png', dpi=300, bbox_inches='tight')

    # Plot the mean of the postage stamps
    psf_guess = guess_psf(ps_exposure, ps_weight, ps_mask)
    vmin, vmax = np.percentile(np.abs(psf_guess), [1., 99.9])

    fig = plt.figure(figsize=(6,6), dpi=200)
    ax = fig.add_subplot(1,1,1)

    ax.imshow(psf_guess.T, origin='upper', aspect='equal',
              interpolation='nearest', cmap='bwr_r',
              vmin=-vmax, vmax=vmax)

    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig('psf_guess.png', dpi=300, bbox_inches='tight')

    # Fit a flux and sky brightness for each star
    idx = filter_postage_stamps(ps_mask, min_pixel_fraction=0.5)
    #ps_exposure = ps_exposure[idx]
    #ps_weight = ps_weight[idx]
    #ps_mask = ps_mask[idx]
    #star_x = star_x[idx]
    #star_y = star_y[idx]

    n_ps = ps_exposure.shape[0]

    psf_coeffs = np.zeros((6, ps_exposure.shape[1], ps_exposure.shape[2]), dtype='f8')
    psf_coeffs[0,:,:] = psf_guess[:,:]
    a0 = np.zeros(n_ps, dtype='f8')
    a1 = np.zeros(n_ps, dtype='f8')

    for k in range(n_ps):
        if not idx[k]:
            continue

        a0[k], a1[k] = fit_star_params(psf_coeffs, star_x[k], star_y[k],
                                       ps_exposure[k], ps_weight[k], ps_mask[k],
                                       ccd_shape)

    # Plot the residuals of each star
    n_x = int(np.ceil(np.sqrt(n_ps)))
    n_y = int(np.ceil(n_ps/float(n_x)))

    fig = plt.figure(figsize=(n_x,n_y), dpi=100)

    for k in range(n_ps):
        #tmp = a0[k] * psf_guess + a1[k]
        tmp = ps_exposure[k]
        tmp[ps_mask[k] != 0] = np.nan

        vmax = 1.

        idx = (tmp > 1.)
        if np.any(idx):
            vmax = np.percentile(np.abs(tmp[idx]), 99.5)

        tmp = ps_exposure[k] - a0[k] * psf_guess - a1[k]
        tmp[ps_mask[k] != 0] = np.nan

        ax = fig.add_subplot(n_x, n_y, k+1, axisbg='g')

        ax.imshow(tmp.T, origin='upper', aspect='equal',
                  interpolation='nearest', cmap='bwr_r',
                  vmin=-vmax, vmax=vmax)

        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    fig.savefig('postage_stamp_resids.png', dpi=300, bbox_inches='tight')

    # Fit PSF model from stars
    psf_coeffs = fit_psf_coeffs(a0, a1, star_x, star_y, img_data.shape,
                                ps_exposure, ps_weight, ps_mask)
    psf_img = eval_psf(psf_coeffs, 0.5, 0.5, (1.,1.))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    vmax = np.percentile(np.abs(psf_img[np.isfinite(psf_img)]), 99.9)
    ax.imshow(psf_img.T, origin='upper', aspect='equal',
              interpolation='nearest', cmap='bwr_r',
              vmin=-vmax, vmax=vmax)

    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig('psf_fit.png', dpi=300, bbox_inches='tight')

    #plt.show()


def test_filter_neighbors():
    n = 50
    r = 0.05
    x = np.random.random(n)
    y = np.random.random(n)

    idx = filter_close_stars(x, y, r=r)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(x[idx], y[idx], c='b')
    ax.scatter(x[~idx], y[~idx], c='r')

    plt.show()


def test_extract_psf():
    # Load a test image
    fname_pattern = 'psftest/c4d_150109_051822_oo{}_z_v1.fits.fz'
    img_data, weight_data, mask_data, wcs = load_exposure(fname_pattern, 'S31')
    ccd_shape = img_data.shape

    print 'Weight percentiles:'
    print np.percentile(weight_data[np.isfinite(weight_data) & (mask_data == 0)], [0.1, 1., 5., 50., 95., 99., 99.9])
    print ''

    # Extract the PSF
    psf_coeffs, star_dict = extract_psf(img_data, weight_data, mask_data, wcs,
                                        return_postage_stamps=True, n_iter=1,
                                        min_pixel_fraction=0.9999)

    # Calculate centers of mass
    #print 'Centers of mass:'

    #com_xy = []
    #com_filt_xy = []

    #width = int(np.round((star_dict['ps_exposure'].shape[1] - 1.) / 2.))
    #w_cent = gen_centered_weights(star_dict['ps_exposure'].shape[1:], 1.)

    #for ps_img, ps_mask in zip(star_dict['ps_exposure'], star_dict['ps_mask']):
    #    com_xy.append(img_center_of_mass(ps_img, ps_mask))
    #    com_filt_xy.append(img_center_of_mass(ps_img*w_cent, ps_mask))
    #    print com_xy[-1], com_filt_xy[-1]

    #print ''

    # Calculate the residuals
    psf_resid = calc_star_chisq(psf_coeffs, star_dict['ps_exposure'],
                                star_dict['ps_weight'], star_dict['ps_mask'],
                                star_dict['star_x'], star_dict['star_y'],
                                star_dict['stellar_flux'], star_dict['sky_level'],
                                ccd_shape)

    import matplotlib.pyplot as plt

    # Scatterplot of PS1 flux with inferred flux
    fig = plt.figure(figsize=(16,8), dpi=100)

    ps1_flux = 10.**(-(star_dict['star_ps1_mag']-20.) / 2.5)
    fit_mag = -2.5 * np.log10(star_dict['stellar_flux'])

    idx_ps1_good = (star_dict['star_ps1_mag'][:,3] > 10.) & (star_dict['star_ps1_mag'][:,3] < 25.)
    mag_offset = np.median(fit_mag[idx_ps1_good] - star_dict['star_ps1_mag'][idx_ps1_good,3])
    flux_scaling = np.median(star_dict['stellar_flux'][idx_ps1_good] / ps1_flux[idx_ps1_good,3])

    print 'Magnitude offset: {}'.format(mag_offset)
    print 'Flux scaling: {}'.format(flux_scaling)

    ax = fig.add_subplot(1,2,1)
    ax.scatter(ps1_flux[idx_ps1_good,3], star_dict['stellar_flux'][idx_ps1_good],
               edgecolor='none', facecolor='r', s=10)
    ax.set_xlabel(r'$f_{\mathrm{P1}}$', fontsize=18)
    ax.set_ylabel(r'$f_{\mathrm{fit}}$', fontsize=18)
    ax.set_title(r'$\mathrm{Fit \ vs. \ PS1 \ Flux}$', fontsize=20)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_line = [xlim[0] * flux_scaling, xlim[1] * flux_scaling]
    ax.plot(xlim, y_line, ls='-', lw=2., alpha=0.25, c='r')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = fig.add_subplot(1,2,2)
    ax.scatter(star_dict['star_ps1_mag'][idx_ps1_good,3], fit_mag[idx_ps1_good],
               edgecolor='none', facecolor='r', s=10)
    ax.set_xlabel(r'$m_{\mathrm{P1}}$', fontsize=18)
    ax.set_ylabel(r'$m_{\mathrm{fit}}$', fontsize=18)
    ax.set_title(r'$\mathrm{Fit \ vs. \ PS1 \ Magnitude}$', fontsize=20)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_line = [xlim[0] + mag_offset, xlim[1] + mag_offset]
    ax.plot(xlim, y_line, ls='-', lw=2., alpha=0.25, c='r')

    ax.set_xlim(xlim[::-1])
    ax.set_ylim(ylim[::-1])

    fig.savefig('star_fit_vs_ps1.svg', bbox_inches='tight')
    plt.close(fig)

    # Scatterplot of fit flux vs. sky level
    fig = plt.figure(figsize=(8,8), dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.scatter(fit_mag, star_dict['sky_level'],
               edgecolor='none', facecolor='b', s=10)
    ax.set_xlabel(r'$\mathrm{fit \ magnitude}$', fontsize=18)
    ax.set_ylabel(r'$\mathrm{sky \ level}$', fontsize=18)
    ax.set_title(r'$\mathrm{Fit \ Parameters}$', fontsize=20)

    fig.savefig('star_fit_params.svg', bbox_inches='tight')
    plt.close(fig)

    # Plot the PSF coefficients
    fig = plt.figure(figsize=(10,6), dpi=100)
    psf_vmax = np.max(np.abs(psf_coeffs))

    order_label = [r'$1$', r'$x$', r'$y$', r'$x^2$', r'$y^2$', r'$xy$']

    for k in range(6):
        ax = fig.add_subplot(2,3,k+1)

        ax.imshow(psf_coeffs[k].T, origin='upper', aspect='equal',
                  interpolation='nearest', cmap='bwr_r',
                  vmin=-psf_vmax, vmax=psf_vmax)

        ax.set_title(order_label[k], fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig('psf_coeffs.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Plot the PSF across the CCD
    fig = plt.figure(figsize=(12,12), dpi=120)

    for j,x in enumerate(np.linspace(0, ccd_shape[0], 3)):
        for k,y in enumerate(np.linspace(0, ccd_shape[1], 3)):
            tmp = eval_psf(psf_coeffs, x, y, ccd_shape)

            ax = fig.add_subplot(3,3,3*k+j+1)

            ax.imshow(tmp.T, origin='upper', aspect='equal',
                      interpolation='nearest', cmap='bwr_r',
                      vmin=-psf_vmax, vmax=psf_vmax)

            ax.set_xticks([])
            ax.set_yticks([])

    fig.subplots_adjust(wspace=0.01, hspace=0.01)

    fig.savefig('psf_over_ccd.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Plot postage stamps of the stars
    n_stars = star_dict['ps_exposure'].shape[0]
    ccd_shape = img_data.shape
    ps_x_cent = 0.5 * float(star_dict['ps_exposure'].shape[1]-1.)
    ps_y_cent = 0.5 * float(star_dict['ps_exposure'].shape[2]-1.)

    n_x = int(np.ceil(np.sqrt(n_stars)))
    n_y = int(np.ceil(n_stars/float(n_x)))

    fig = plt.figure(figsize=(n_x,n_y), dpi=100)

    ps_vmax = []

    for k in range(n_stars):
        tmp = star_dict['ps_exposure'][k]
        tmp[star_dict['ps_mask'][k] != 0] = np.nan

        ps_vmax.append(1.)

        idx = (tmp > 1.)
        if np.any(idx):
            ps_vmax[-1] = np.percentile(np.abs(tmp[idx]), 99.5)

        ax = fig.add_subplot(n_x, n_y, k+1, axisbg='g')

        ax.imshow(tmp.T, origin='upper', aspect='equal',
                  interpolation='nearest', cmap='bwr_r',
                  vmin=-ps_vmax[-1], vmax=ps_vmax[-1])

        ax.scatter([ps_x_cent], [ps_y_cent], s=3., edgecolor='none',
                                             facecolor='cyan', alpha=0.5)

        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(wspace=0.01, hspace=0.01)

    fig.savefig('postage_stamps.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot postage stamps of the residuals (after subtracting off model of psf)
    fig = plt.figure(figsize=(n_x,n_y), dpi=100)
    fig_stretch = plt.figure(figsize=(n_x,n_y), dpi=100)

    tmp = star_dict['ps_exposure']
    tmp = tmp[(star_dict['ps_mask'] == 0) & np.isfinite(tmp)]
    tmp_pctiles = np.percentile(tmp, [10., 90.])
    tmp = tmp[(tmp > tmp_pctiles[0]) & (tmp < tmp_pctiles[1])]
    vmax_stretch = 4. * np.std(tmp)

    for k in range(n_stars):
        # Evaluate the PSF at the location of the star
        psf_model = eval_psf(psf_coeffs, star_dict['star_x'][k],
                             star_dict['star_y'][k], ccd_shape)

        #print ''
        #print 'norm = {:5f}'.format(np.sum(psf_model))

        psf_model *= star_dict['stellar_flux'][k] # Multiply by the star's flux
        psf_model += star_dict['sky_level'][k] # Add in the sky level

        #print psf_coeffs[0,28:36,28:36]
        #print star_dict['stellar_flux'][k]
        #print star_dict['sky_level'][k]

        # Calculate the residuals and apply the mask
        tmp = star_dict['ps_exposure'][k] - psf_model
        tmp[star_dict['ps_mask'][k] != 0] = np.nan

        ax = fig.add_subplot(n_x, n_y, k+1, axisbg='g')
        ax_stretch = fig_stretch.add_subplot(n_x, n_y, k+1, axisbg='g')

        ax.imshow(tmp.T, origin='upper', aspect='equal',
                  interpolation='nearest', cmap='bwr_r',
                  vmin=-ps_vmax[k], vmax=ps_vmax[k])
        ax_stretch.imshow(tmp.T, origin='upper', aspect='equal',
                  interpolation='nearest', cmap='bwr_r',
                  vmin=-vmax_stretch, vmax=vmax_stretch)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_txt = xlim[0] + 0.05 * (xlim[1] - xlim[0])
        y_txt = ylim[0] + 0.05 * (ylim[1] - ylim[0])

        ax.text(x_txt, y_txt, r'${:.1f}$'.format(psf_resid[k]))
        ax_stretch.text(x_txt, y_txt, r'${:.1f}$'.format(psf_resid[k]))

        ax.scatter([ps_x_cent], [ps_y_cent], s=3., edgecolor='none',
                                             facecolor='cyan', alpha=0.5)
        ax_stretch.scatter([ps_x_cent], [ps_y_cent], s=3., edgecolor='none',
                                             facecolor='cyan', alpha=0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax_stretch.set_xticks([])
        ax_stretch.set_yticks([])

    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig_stretch.subplots_adjust(wspace=0.01, hspace=0.01)

    fig.savefig('postage_stamps_resid.png', dpi=300, bbox_inches='tight')
    fig_stretch.savefig('postage_stamps_resid_stretch.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close(fig_stretch)

    # Plot the CCD image and weight, with stellar locations from PS1 overplotted
    vmin, vmax = np.percentile(img_data[(img_data > 1.) & (mask_data == 0)], [1.,99.])

    import matplotlib.pyplot as plt
    fig = plt.figure(dpi=300)

    ax = fig.add_subplot(2,1,1, axisbg='g')
    img = np.copy(img_data)
    img[mask_data != 0] = np.nan
    ax.imshow(img.T, origin='upper', aspect='equal', interpolation='nearest',
                          cmap='binary_r', vmin=vmin, vmax=vmax)

    ax.scatter(star_dict['star_x'], star_dict['star_y'],
               s=12, edgecolor='b', facecolor='none',
               lw=0.75, alpha=0.75)

    ax.set_xlim(0, img_data.shape[0])
    ax.set_ylim(0, img_data.shape[1])

    ax = fig.add_subplot(2,1,2, axisbg='g')
    sigma = 1./np.sqrt(weight_data)
    sigma[mask_data != 0] = np.nan
    vmin_s, vmax_s = np.percentile(sigma[np.isfinite(sigma) & (mask_data == 0)], [1., 99.])
    ax.imshow(sigma.T, origin='upper', aspect='equal', interpolation='nearest',
                       cmap='binary_r', vmin=vmin_s, vmax=vmax_s)

    ax.scatter(star_dict['star_x'], star_dict['star_y'],
               s=12, edgecolor='b', facecolor='none',
               lw=0.75, alpha=0.75)

    ax.set_xlim(0, img_data.shape[0])
    ax.set_ylim(0, img_data.shape[1])

    fig.savefig('ccd_with_stars.png', dpi=300, bbox_inches='tight')



def main():
    test_extract_psf()

    return 0


if __name__ == '__main__':
    main()
