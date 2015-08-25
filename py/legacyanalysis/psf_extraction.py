#!/usr/env python

import numpy as np
import scipy.ndimage

import astropy.wcs
import astropy.io


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


def extract_star(exposure_img, error_img, wcs, star_world_coords, width):
    pix_center = wcs.wcs_world2pix(star_world_coords, 0)   # 0 is the coordinate in the top left (the numpy, but not FITS standard)


    return pix_center


def filter_ps1(ps1_table):
    '''
    Return a filtered copy of a table of PS1 detections, selecting compact
    sources detected in multiple exposures.
    '''

    idx = ((np.sum(ps1_table['NMAG_OK'], axis=1) >= 5) &
           (np.sum(ps1_table['MEAN'] - ps1_table['MEAN_AP'] < 0.1, axis=1) >= 2))

    return ps1_table[idx]


def extract_psf(exposure_img, error_img, wcs, star_world_coords):
    # select stars (from PS1)

    # center stars (using sinc shift)
    # standardize brightness of each star (either using PS1 or by scaling to median)
    # In loop:
    #   solve linear equation to minimize residuals between model and data
    #     --> reject saturated pixels
    #   update brightness of each star
    pass


def test_sinc_shift_image():
    from PIL import Image
    im = Image.open('/home/greg/Downloads/download.png')
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
    Load one CCD from an exposure. Zero out the weights of bad pixels (as
    defined by the mask).

    Inputs:
      fname_pattern  A filename of the form c4d_150109_051822_oo{}_z_v1.fits.fz,
                     where {} will be expanded in order to select the image,
                     weight and mask files.
      ccd_id         The identifier of the desired CCD (e.g., 'S31').

    Outputs:
      img_data     Exposure image
      weight_data  Weight image
      wcs          The WCS astrometric solution
    '''

    img_data, img_header = astropy.io.fits.getdata(fname_pattern.format('i'),
                                                   ccd_id, header=True)
    wcs = astropy.wcs.WCS(header=img_header)

    weight_data = astropy.io.fits.getdata(fname_pattern.format('w'), ccd_id)
    mask_data = astropy.io.fits.getdata(fname_pattern.format('d'), ccd_id)

    # Apply the mask to the weights (and zero out the corresponding image pixels)
    mask_idx = (mask_data != 0)
    img_data[mask_idx] = 0.
    weight_data[mask_idx] = 0.

    return img_data, weight_data, wcs


def test_find_star_centers():
    pass


def test_load_exposure():
    # Load a test image
    fname_pattern = '/home/greg/Downloads/psftest/c4d_150109_051822_oo{}_z_v1.fits.fz'
    img_data, weight_data, wcs = load_exposure(fname_pattern, 'S31')

    # Load locations of PS1 stars
    fname = '/home/greg/Downloads/psftest/ps1stars-c4d_150109_051822.fits'
    ps1_table = astropy.io.fits.getdata(fname, 1)
    ps1_table = filter_ps1(ps1_table)
    star_y, star_x = wcs.wcs_world2pix(ps1_table['RA'], ps1_table['DEC'], 0)

    # Crop stars to CCD boundaries
    idx = (star_x > 0.) & (star_x < img_data.shape[0]) & (star_y > 0.) & (star_y < img_data.shape[1])
    star_x = star_x[idx]
    star_y = star_y[idx]

    # Plot the CCD image and weight, with stellar locations from PS1 overplotted
    vmin, vmax = np.percentile(img_data[img_data > 1.], [1.,99.])

    import matplotlib.pyplot as plt
    fig = plt.figure(dpi=300)

    ax = fig.add_subplot(2,1,1)
    ax.imshow(img_data.T, origin='upper', aspect='equal', interpolation='nearest',
                        cmap='binary_r', vmin=vmin, vmax=vmax)

    ax.scatter(star_x, star_y, s=12, edgecolor='g', facecolor='none',
                               lw=0.5, alpha=0.75)

    ax.set_xlim(0, img_data.shape[0])
    ax.set_ylim(0, img_data.shape[1])

    ax = fig.add_subplot(2,1,2)
    sigma = 1./np.sqrt(weight_data)
    vmin, vmax = np.percentile(sigma[np.isfinite(sigma)], [1., 99.])
    ax.imshow(sigma.T, origin='upper', aspect='equal', interpolation='nearest',
                     cmap='binary_r', vmin=vmin, vmax=vmax)

    fig.savefig('ccd_with_ps1_detections.png', dpi=300, bbox_inches='tight')

    plt.show()


def main():
    test_load_exposure()

    #fname_pattern = '/home/greg/Downloads/psftest/c4d_150109_051822_oo{}_z_v1.fits.fz'
    #img_data, weight_data, wcs = load_exposure(fname_pattern, 'S31')


    return 0


if __name__ == '__main__':
    main()
