import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import astropy.convolution

from psf_extraction import *


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


def gen_test_data(psf_coeffs, sky_level=100., mag_20_flux=100., band=3):
    psf_shape = psf_coeffs.shape[1:]

    # Load a test image
    fname_pattern = 'psftest/c4d_150109_051822_oo{}_z_v1.fits.fz'
    img_data, weight_data, mask_data, wcs = load_exposure(fname_pattern, 'S31')
    ccd_shape = img_data.shape

    # Load locations of PS1 stars
    fname = 'psftest/ps1stars-c4d_150109_051822.fits'
    ps1_table = astropy.io.fits.getdata(fname, 1)
    star_x, star_y, ps1_mag = get_star_locations(ps1_table, wcs, img_data.shape, min_separation=1.)

    # Clip to stars that fall on the CCD
    #idx = ( (star_x > -psf_shape[0]) & (star_x < ccd_shape[0]+psf_shape[0])
    #      & (star_y > -psf_shape[1]) & (star_y < ccd_shape[1]+psf_shape[1]) )

    #idx = np.arange(10)
    #star_x = star_x[idx]
    #star_y = star_y[idx]
    #ps1_table = ps1_table[idx]

    # Transform stellar magnitudes to fluxes
    star_flux = mag_20_flux * 10.**(-(ps1_mag[:,band]-20.)/5.)

    # Work on a larger canvas, so that edge effects can be ignored
    img_shape = [ccd_shape[0]+2*psf_shape[0], ccd_shape[1]+2*psf_shape[1]]
    dx = 0.5 * (psf_shape[0] - 1)
    dy = 0.5 * (psf_shape[1] - 1)

    # Fill the image with Gaussian background noise
    img_mean = np.ones(shape=img_shape, dtype='f8')
    #img_mock = np.random.normal(loc=sky_level, scale=np.sqrt(sky_level), size=img_data.shape)

    psf_workspace = np.empty(shape=img_shape, dtype='f8')

    # Add in each star
    for k in range(len(star_x)):
        print k, star_flux[k], ps1_mag[k,band]

        # Evaluate the PSF at the position of the star
        #psf_img = eval_psf(psf_coeffs, star_x[k], star_y[k], ccd_shape)
        psf_workspace[:] = 0.
        psf_workspace[:psf_shape[0],:psf_shape[1]] = eval_psf(psf_coeffs, star_x[k], star_y[k], ccd_shape)
        psf_workspace *= star_flux[k] / np.sum(psf_workspace)

        # Shift the PSF to be centered on the star
        x_shift = int(np.around(star_x[k]+dx))
        y_shift = int(np.around(star_y[k]+dy))
        psf_workspace = np.roll(psf_workspace, x_shift, axis=0)
        psf_workspace = np.roll(psf_workspace, y_shift, axis=1)

        img_mean += psf_workspace
        #img_mean += sinc_shift_image(psf_workspace, star_x[k]+dx, star_y[k]+dy)

    # Clip the image
    img_mean = img_mean[psf_shape[0]:-psf_shape[0], psf_shape[1]:-psf_shape[1]]

    # Add noise into the image
    img_model = img_mean + np.sqrt(img_mean) * np.random.normal(loc=0., scale=1., size=ccd_shape)

    # Calculate the weights (assume Poisson statistics)
    weight_model = 1. / img_mean

    # Copy the mask
    mask_model = np.empty(ccd_shape, dtype='f8')
    mask_model[:] = mask_data[:]

    return img_model, weight_model, mask_model, star_x, star_y


def test_gen_data():
    psf_sigma = 3.
    psf_coeffs = np.zeros((6,63,63), dtype='f8')
    psf_coeffs[0] = np.array(astropy.convolution.Gaussian2DKernel(
        psf_sigma, x_size=psf_coeffs.shape[1], y_size=psf_coeffs.shape[2],
        mode='oversample', factor=5
    ))

    img_model, weight_model, mask_model, star_x, star_y  = gen_test_data(psf_coeffs)

    # Plot the CCD image and weight, with stellar locations from PS1 overplotted
    vmin, vmax = np.percentile(img_model[(img_model > 1.) & (mask_model == 0)], [1.,99.])

    fig = plt.figure(dpi=300)

    ax = fig.add_subplot(2,1,1, axisbg='g')
    img_model[(mask_model != 0)] = np.nan
    ax.imshow(img_model.T, origin='upper', aspect='equal', interpolation='nearest',
                           cmap='binary_r', vmin=vmin, vmax=vmax)

    ax.scatter(star_x, star_y,
               s=12, edgecolor='b', facecolor='none',
               lw=0.75, alpha=0.75)

    ax.set_xlim(0, img_model.shape[0])
    ax.set_ylim(0, img_model.shape[1])

    ax = fig.add_subplot(2,1,2, axisbg='g')
    sigma = 1. / np.sqrt(weight_model)
    vmin_s, vmax_s = np.percentile(sigma[np.isfinite(sigma) & (mask_model == 0)], [0.1, 99.5])
    sigma[(mask_model != 0)] = np.nan
    ax.imshow(sigma.T, origin='upper', aspect='equal', interpolation='nearest',
                       cmap='binary_r', vmin=vmin_s, vmax=vmax_s)

    fig.savefig('mock_data/ccd_model.png', dpi=300, bbox_inches='tight')

    return



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

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(x[idx], y[idx], c='b')
    ax.scatter(x[~idx], y[~idx], c='r')

    plt.show()


def test_extract_psf():
    # Load a test image
    #fname_pattern = 'psftest/c4d_150109_051822_oo{}_z_v1.fits.fz'
    fname_pattern = 'psfunit/detest-00396086-S31-oo{}.fits.fz'
    ccd_id = 'S31'
    img_data, weight_data, mask_data, wcs, exp_num = load_exposure(fname_pattern, ccd_id)
    ccd_shape = img_data.shape

    # Extract the PSF
    psf_coeffs, star_dict = extract_psf(img_data, weight_data, mask_data, wcs,
                                        return_postage_stamps=True, n_iter=5,
                                        min_pixel_fraction=0.75,
                                        star_chisq_threshold=2.)

    n_stars = star_dict['ps_exposure'].shape[0]
    ccd_shape = img_data.shape

    # Write the PSF to a file
    psf_fname = 'decam-{:08d}-{}.fits'.format(exp_num, ccd_id)
    write_psf_file(psf_fname, psf_coeffs, ccd_shape, ccd_id, exp_num)

    #m_tmp = np.arange(14., 20.1, 1.)
    #f_tmp = flux_predictor(m_tmp)

    #print ''
    #for m,f in zip(m_tmp, f_tmp):
    #    print 'f({:d}) = {.5f}'.format(m, f)
    #print ''

    # Calculate the residuals
    psf_resid = calc_star_chisq(psf_coeffs, star_dict['ps_exposure'],
                                star_dict['ps_weight'], star_dict['ps_mask'],
                                star_dict['star_x'], star_dict['star_y'],
                                star_dict['stellar_flux'], star_dict['sky_level'],
                                ccd_shape)

    # Calculate the shift for each star
    print 'Fitting star offsets:'
    for k in range(len(star_dict['ps_exposure'])):
        print 'Star {: 3d}'.format(k)
        print '========\n'

        dx,dy = fit_star_offset(psf_coeffs, star_dict['ps_exposure'][k],
                                star_dict['ps_weight'][k], star_dict['ps_mask'][k],
                                star_dict['star_x'][k], star_dict['star_y'][k],
                                star_dict['stellar_flux'][k], star_dict['sky_level'][k],
                                ccd_shape, max_shift=5.)

        print 'dx,dy = ({:.3f}, {:.3f})\n\n'.format(dx, dy)

    # Calculate corrections to fluxes, by normalizing PSFs
    psf_norm = np.empty(n_stars, dtype='f8')

    for k in range(n_stars):
        # Evaluate the PSF at the location of the star
        psf_model = eval_psf(psf_coeffs, star_dict['star_x'][k],
                             star_dict['star_y'][k], ccd_shape)
        psf_norm[k] = np.sum(psf_model)

    #star_dict['stellar_flux'] *= psf_norm

    # Scatterplot of PS1 flux with inferred flux
    fig = plt.figure(figsize=(16,8), dpi=100)

    ps1_flux = 10.**(-(star_dict['star_ps1_mag']-20.) / 2.5)
    fit_flux = star_dict['stellar_flux'] * psf_norm
    fit_mag = -2.5 * np.log10(fit_flux)

    idx_ps1_good = (star_dict['star_ps1_mag'][:,3] > 10.) & (star_dict['star_ps1_mag'][:,3] < 25.)
    mag_offset = np.median(fit_mag[idx_ps1_good] - star_dict['star_ps1_mag'][idx_ps1_good,3])
    flux_scaling = np.median(fit_flux[idx_ps1_good] / ps1_flux[idx_ps1_good,3])

    # Relation between PS1 flux and fitted flux
    flux_predictor = gen_stellar_flux_predictor(star_dict['stellar_flux'],
                                                star_dict['star_ps1_mag'],
                                                star_dict['star_x'],
                                                star_dict['star_y'],
                                                ccd_shape,
                                                psf_coeffs)

    print 'Magnitude offset: {}'.format(mag_offset)
    print 'Flux scaling: {}'.format(flux_scaling)

    ax = fig.add_subplot(1,2,1)

    ax.scatter(ps1_flux[idx_ps1_good,3], fit_flux[idx_ps1_good],
               edgecolor='none', facecolor='r', s=10)


    idx = np.all((star_dict['star_ps1_mag'] > 1.) & (star_dict['star_ps1_mag'] < 26.), axis=1)
    f_tmp = ps1_flux[idx]
    m_tmp = star_dict['star_ps1_mag'][idx]
    print m_tmp[:3]

    print 'resid:'
    print (flux_predictor(m_tmp) - fit_flux[idx]) / fit_flux[idx]
    print ''

    ax.scatter(f_tmp[:,3], flux_predictor(m_tmp),
               edgecolor='none', facecolor='g', s=10, alpha=0.5)

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
    for vmax in [1., 0.1, 0.01, 0.001]:
        fig = plt.figure(figsize=(12,12), dpi=120)

        for j,x in enumerate(np.linspace(0, ccd_shape[0], 3)):
            for k,y in enumerate(np.linspace(0, ccd_shape[1], 3)):
                tmp = eval_psf(psf_coeffs, x, y, ccd_shape)
                tmp /= np.max(np.abs(tmp))

                ax = fig.add_subplot(3,3,3*k+j+1)

                ax.imshow(tmp.T, origin='upper', aspect='equal',
                          interpolation='nearest', cmap='bwr_r',
                          vmin=-vmax, vmax=vmax)

                ax.set_xticks([])
                ax.set_yticks([])

        fig.subplots_adjust(wspace=0.01, hspace=0.01)

        fig.savefig('psf_over_ccd_{:.3f}.png'.format(vmax),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    # Plot the central PSF shifted by different amounts
    psf_img = eval_psf(psf_coeffs, 0.5, 0.5, (1.,1.))
    psf_img /= np.max(psf_img)

    for vmax in [1., 0.1, 0.01, 0.001]:
        fig = plt.figure(figsize=(12,12), dpi=120)

        for j,dx in enumerate(np.linspace(-5., 5., 3)):
            for k,dy in enumerate(np.linspace(-5., 5., 3)):
                tmp = sinc_shift_image(psf_img, dx, dy)#, roll_int=False)

                ax = fig.add_subplot(3,3,3*k+j+1)

                ax.imshow(tmp.T, origin='upper', aspect='equal',
                          interpolation='nearest', cmap='bwr_r',
                          vmin=-vmax, vmax=vmax)

                ax.set_xticks([])
                ax.set_yticks([])

        fig.subplots_adjust(wspace=0.01, hspace=0.01)

        fig.savefig('psf_shifted_{:.3f}.png'.format(vmax),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    # Plot postage stamps of the stars
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

        tmp = (star_dict['ps_chisq_mask'][k] == 0).astype('f8')
        tmp[tmp < 0.01] = np.nan

        ax.imshow(tmp.T, origin='upper', aspect='equal',
                  interpolation='nearest', cmap='winter',
                  vmin=0., vmax=1., alpha=0.25)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.scatter([ps_x_cent], [ps_y_cent], s=3., edgecolor='none',
                                             facecolor='cyan', alpha=0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.subplots_adjust(wspace=0.01, hspace=0.01)

    fig.savefig('postage_stamps.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot postage stamps of the weights for each star
    fig = plt.figure(figsize=(n_x,n_y), dpi=100)

    for k in range(n_stars):
        tmp = star_dict['ps_weight'][k] * star_dict['ps_exposure'][k]
        tmp[star_dict['ps_mask'][k] != 0] = np.nan

        vmin, vmax = 0., 1.

        idx = np.isfinite(tmp)
        if np.any(idx):
            #vmin, vmax = np.percentile(tmp[idx], [1., 99.8])
            vmin, vmax = np.min(tmp[idx]), np.max(tmp[idx])

        ax = fig.add_subplot(n_x, n_y, k+1, axisbg='g')

        ax.imshow(tmp.T, origin='upper', aspect='equal',
                  interpolation='nearest', cmap='bwr_r',
                  vmin=vmin, vmax=vmax)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.scatter([ps_x_cent], [ps_y_cent], s=3., edgecolor='none',
                                             facecolor='cyan', alpha=0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.subplots_adjust(wspace=0.01, hspace=0.01)

    fig.savefig('postage_stamp_weights.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot postage stamps of the residuals (after subtracting off model of psf)
    fig = plt.figure(figsize=(n_x,n_y), dpi=100)
    fig_stretch = plt.figure(figsize=(n_x,n_y), dpi=100)
    fig_weighted = plt.figure(figsize=(n_x,n_y), dpi=100)

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

        chi = tmp * np.sqrt(star_dict['ps_weight'][k])

        ax = fig.add_subplot(n_x, n_y, k+1, axisbg='g')
        ax_stretch = fig_stretch.add_subplot(n_x, n_y, k+1, axisbg='g')
        ax_weighted = fig_weighted.add_subplot(n_x, n_y, k+1, axisbg='g')

        ax.imshow(tmp.T, origin='upper', aspect='equal',
                  interpolation='nearest', cmap='bwr_r',
                  vmin=-ps_vmax[k], vmax=ps_vmax[k])
        ax_stretch.imshow(tmp.T, origin='upper', aspect='equal',
                  interpolation='nearest', cmap='bwr_r',
                  vmin=-vmax_stretch, vmax=vmax_stretch)
        ax_weighted.imshow(chi.T, origin='upper', aspect='equal',
                  interpolation='nearest', cmap='bwr_r',
                  vmin=-3., vmax=3.)

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
        ax_weighted.scatter([ps_x_cent], [ps_y_cent], s=3., edgecolor='none',
                                             facecolor='cyan', alpha=0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax_stretch.set_xticks([])
        ax_stretch.set_yticks([])
        ax_weighted.set_xticks([])
        ax_weighted.set_yticks([])

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax_stretch.set_xlim(xlim)
        ax_stretch.set_ylim(ylim)
        ax_weighted.set_xlim(xlim)
        ax_weighted.set_ylim(ylim)

    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig_stretch.subplots_adjust(wspace=0.01, hspace=0.01)
    fig_weighted.subplots_adjust(wspace=0.01, hspace=0.01)

    fig.savefig('postage_stamps_resid.png', dpi=300, bbox_inches='tight')
    fig_stretch.savefig('postage_stamps_resid_stretch.png', dpi=300, bbox_inches='tight')
    fig_weighted.savefig('postage_stamps_resid_weighted.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close(fig_stretch)
    plt.close(fig_weighted)

    # Plot the CCD image and weight, with stellar locations from PS1 overplotted
    vmin, vmax = np.percentile(img_data[(img_data > 1.) & (mask_data == 0)], [1.,99.])

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
    #test_extract_psf()
    test_gen_data()

    return 0


if __name__ == '__main__':
    main()
