import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import astropy.convolution

from psf_extraction import *


def gen_mock_postage_stamps(psf_coeffs, n_stars, flux_max=100.,
                            mask_prob=0.25, max_mask_size=0.25,
                            x_star=None, y_star=None):
        # Set up the PSF
        assert (psf_coeffs.shape[1] == psf_coeffs.shape[2])
        psf_size = psf_coeffs.shape[1]
        psf_coeffs = normalize_psf_coeffs(psf_coeffs)

        # Generate the stellar postage stamps
        ccd_shape = (4000,2000)
        flux_background = 10.

        if x_star == None:
            x_star = ccd_shape[0] * np.random.random(n_stars)
        if y_star == None:
            y_star = ccd_shape[1] * np.random.random(n_stars)

        flux_model = flux_max * flux_background * np.random.random(n_stars)
        sky_model = np.random.normal(loc=flux_background,
                                     scale=np.sqrt(flux_background),
                                     size=n_stars)

        ps_img = np.empty((n_stars,psf_size,psf_size), dtype='f8')

        for k,(x,y) in enumerate(zip(x_star, y_star)):
            psf_local = eval_psf(psf_coeffs, x, y, ccd_shape)
            psf_local /= np.sum(psf_local)
            ps_img[k] = sky_model[k] + flux_model[k] * psf_local

        sigma2 = np.abs(ps_img)     # Poisson statistics
        sigma2[sigma2 < 1.] = 1.    # Floor on the variance

        ps_weight = 1. / sigma2
        ps_mask = np.zeros(ps_img.shape, dtype='f8')    # No masked pixels

        idx_star_mask = np.where(np.random.random(n_stars) < mask_prob)[0]
        for k in idx_star_mask:
            w = psf_size * max_mask_size * np.random.random()
            i0 = int(np.floor(0.5 * (psf_size - w)))
            i1 = int(np.ceil(0.5 * (psf_size + w))) + 1
            ps_mask[k,i0:i1,i0:i1] = 1.

        # Add in Poisson noise to the image
        ps_img += np.sqrt(sigma2) * np.random.normal(loc=0., scale=1., size=ps_img.shape)

        return ps_img, ps_weight, ps_mask, x_star, y_star, flux_model, sky_model, ccd_shape


def test_chisq(add_secondary_sources=False):
    plt_suffix = '_bad' if add_secondary_sources else ''
    title_suffix = r'\ \left( bad \right)' if add_secondary_sources else ''

    # Set up the PSF
    psf_size = 63
    psf_sigma = 3.
    psf_coeffs = np.zeros((6,psf_size,psf_size), dtype='f8')
    psf_coeffs[0] = np.array(astropy.convolution.Gaussian2DKernel(
        psf_sigma, x_size=psf_size, y_size=psf_size,
        mode='oversample', factor=5
    ))
    #psf_coeffs[1] = 0.5 * np.array(astropy.convolution.Gaussian2DKernel(
    #    0.75*psf_sigma, x_size=psf_size, y_size=psf_size,
    #    mode='oversample', factor=5
    #))
    psf_coeffs = normalize_psf_coeffs(psf_coeffs)

    # Generate the stellar postage stamps
    n_stars = 10000
    (ps_img, ps_weight, ps_mask,
     x_star, y_star,
     flux_model, sky_model,
     ccd_shape) = gen_mock_postage_stamps(psf_coeffs, n_stars,
                                          flux_max=1.e10,
                                          mask_prob=0.25,
                                          max_mask_size=0.25)

    # Add in secondary sources
    if add_secondary_sources:
        #res_tmp = gen_mock_postage_stamps(psf_coeffs, n_stars,
        #                                  flux_max=1.e10,
        #                                  mask_prob=0.,
        #                                  max_mask_size=0.,
        #                                  x_star=x_star,
        #                                  y_star=y_star)

        offset = psf_size * (np.random.random((n_stars,2)) - 0.5)

        for k,(dx,dy) in enumerate(offset):
            #print dx, dy
            #mult = 0.1 * flux_model[k] / res_tmp[5][k]
            #ps_img[k] += sinc_shift_image(res_tmp[0][k], dx, dy) * mult
            #s2_tmp = sinc_shift_image(1./res_tmp[1][k], dx, dy) * mult**2.
            #ps_weight[k] = 1. / (1./ps_weight[k] + s2_tmp)
            #ps_img[k] = sinc_shift_image(res_tmp[0][k], dx, dy)
            #ps_weight[k] = w_tmp

            sec_tmp = np.array(astropy.convolution.Gaussian2DKernel(
                psf_sigma, x_size=psf_size, y_size=psf_size,
                mode='oversample', factor=5
            ))
            sec_tmp *= 0.1 * np.random.random() / np.sum(sec_tmp) * flux_model[k]
            sec_tmp = sinc_shift_image(sec_tmp, dx, dy)

            s2_tmp = np.abs(sec_tmp)
            s2_tmp[s2_tmp < 1.e-5] = 1.e-5

            sec_tmp += np.sqrt(s2_tmp) * np.random.normal(loc=0., scale=1., size=sec_tmp.shape)
            ps_weight[k] = 1. / (1./ps_weight[k] + s2_tmp)
            ps_img[k] += sec_tmp


    # Calculate the chi^2 surfaces for each star
    star_chisq, chisq_img = calc_star_chisq(
        psf_coeffs,
        ps_img, ps_weight, ps_mask,
        x_star, y_star,
        flux_model, sky_model,
        ccd_shape,
        return_chisq_img=True
    )

    # Calculate the # of d.o.f. for each star
    star_dof = np.sum(np.sum(np.isfinite(chisq_img), axis=1), axis=1)
    dof_eff = np.mean(star_dof)

    # Plot histogram of chi^2
    fig = plt.figure(figsize=(10,5), dpi=120)

    ax = fig.add_subplot(1,1,1)

    ax.hist(star_chisq, 50, normed=1, histtype='stepfilled')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 1000)
    y = scipy.stats.chi2.pdf(x*dof_eff, dof_eff)
    y *= 0.85 * ylim[1] / np.max(y)
    ax.plot(x, y, 'g-', lw=2., alpha=0.5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(r'$\mathrm{Stellar} \ \chi^2$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{Frequency}$', fontsize=16)
    ax.set_title(r'$\mathrm{{Stellar}} \ \chi^2 \ \mathrm{{Computation \ Test {} }}$'.format(title_suffix),
                 fontsize=18)

    fig.savefig('mock_data/plots/chisq{}_stars_hist.svg'.format(plt_suffix),
                dpi=120, bbox_inches='tight')

    # Plot the chi^2 surfaces
    vmin, vmax = 0., 2.
    fig = plt.figure(figsize=(12,12), dpi=120)
    i = 0
    sort_idx = np.arange(n_stars)#np.argsort(star_chisq)

    for j in range(4):
        for k in range(4):
            ax = fig.add_subplot(4,4,i+1, axisbg='g')


            tmp = chisq_img[sort_idx[i]]
            tmp[tmp > 5.] = np.nan
            #vmin, vmax = np.percentile(np.abs(tmp[np.isfinite(tmp)]), [0.5, 99.5])
            #print vmax
            ax.imshow(tmp.T, origin='lower', aspect='equal',
                      interpolation='none', cmap='bwr',
                      vmin=vmin, vmax=vmax)

            '''
            tmp = np.abs(ps_weight[i][np.isfinite(ps_weight[i])])
            vmin, vmax = np.percentile(tmp, [1., 99.5])
            print vmin, vmax
            ax.imshow(np.log(ps_weight[sort_idx[i]].T),
                      origin='lower', aspect='equal',
                      interpolation='none', cmap='bwr_r',
                      vmin=np.log(vmin), vmax=np.log(vmax))
            '''

            '''
            tmp = np.abs(ps_img[i][np.isfinite(ps_img[i])])
            vmin, vmax = np.percentile(tmp, [1., 99.5])
            ax.imshow(ps_img[sort_idx[i]].T,
                      origin='lower', aspect='equal',
                      interpolation='none', cmap='bwr_r',
                      vmin=-vmax, vmax=vmax)
            '''

            ax.set_xticks([])
            ax.set_yticks([])

            i += 1

    fig.subplots_adjust(hspace=0.02, wspace=0.02)
    fig.suptitle(r'$\chi^2 \ \mathrm{{Surface \ Test {} }}$'.format(title_suffix), fontsize=22, y=0.94)
    fig.savefig('mock_data/plots/chisq{}_surfaces.svg'.format(plt_suffix),
                bbox_inches='tight', dpi=120)

    # Plot chi^2 over the CCD
    fig = plt.figure(figsize=(12,6.5), dpi=120)
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_star, y_star, c=star_chisq,
               cmap='RdYlGn_r', vmin=0.75, vmax=1.25,
               s=15., edgecolor='none', alpha=0.5)
    ax.set_xlim(0, ccd_shape[0])
    ax.set_ylim(0, ccd_shape[1])
    fig.subplots_adjust(top=0.92)
    fig.suptitle(r'$\chi^2 \ \mathrm{{Across \ CCD {} }}$'.format(title_suffix),
                 fontsize=22, y=0.94, va='bottom')
    fig.savefig('mock_data/plots/chisq{}_across_CCD.svg'.format(plt_suffix),
                bbox_inches='tight', dpi=120)


def test_star_fit():
    # Set up the PSF
    psf_size = 63
    psf_sigma = 3.
    psf_coeffs = np.zeros((6,psf_size,psf_size), dtype='f8')
    psf_coeffs[0] = np.array(astropy.convolution.Gaussian2DKernel(
        psf_sigma, x_size=psf_size, y_size=psf_size,
        mode='oversample', factor=5
    ))
    psf_coeffs[1] = 0.5 * np.array(astropy.convolution.Gaussian2DKernel(
        0.75*psf_sigma, x_size=psf_size, y_size=psf_size,
        mode='oversample', factor=5
    ))
    psf_coeffs = normalize_psf_coeffs(psf_coeffs)

    # Generate the stellar postage stamps
    n_stars = 1000
    (ps_img, ps_weight, ps_mask,
     x_star, y_star,
     flux_model, sky_model,
     ccd_shape) = gen_mock_postage_stamps(psf_coeffs, n_stars,
                                          flux_max=1.e4,
                                          mask_prob=0.25,
                                          max_mask_size=0.25)

    # Priors
    sky_fit_sigma = 1.e99
    flux_fit_mean = flux_model
    flux_fit_sigma = 1.e99 * np.sqrt(flux_model)

    # Fit the flux and sky level for each star
    flux_fit = np.empty(n_stars, dtype='f8')
    sky_fit = np.empty(n_stars, dtype='f8')

    for k in range(n_stars):
        flux_fit[k], sky_fit[k] = fit_star_params(
            psf_coeffs,
            x_star[k], y_star[k],
            ps_img[k], ps_weight[k],
            ps_mask[k], ccd_shape,
            sky_sigma=sky_fit_sigma,
            stellar_flux_mean=flux_fit_mean[k],
            stellar_flux_sigma=flux_fit_sigma[k]
        )

    # Plot results
    fig = plt.figure(figsize=(16,7), dpi=120)

    mask_frac = np.sum(np.sum(ps_mask > 1.e-5, axis=1), axis=1) / psf_size**2.

    # True vs. fitted flux
    ax = fig.add_subplot(1,2,1)
    ax.scatter(flux_model, flux_fit, s=5,
               c=mask_frac, vmin=0., vmax=0.1, cmap='RdYlGn_r',
               edgecolor='none', alpha=0.5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = min([xlim[0], ylim[0]])
    x1 = min([xlim[1], ylim[1]])
    ax.plot([x0,x1], [x0,x1], 'g-', alpha=0.2, lw=1.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(r'$\mathrm{True \ Flux}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{Fitted \ Flux}$', fontsize=16)

    # True flux vs. fitted sky level
    ax = fig.add_subplot(1,2,2)
    ax.scatter(flux_model, sky_fit-sky_model,
               c=mask_frac, vmin=0., vmax=0.1, cmap='RdYlGn_r',
               edgecolor='none', alpha=0.5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(xlim, [0.,0.], 'g-', alpha=0.2, lw=1.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(r'$\mathrm{True \ Flux}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{Sky \ Residual}$', fontsize=16)

    fig.suptitle(r'$\mathrm{Stellar \ Parameter \ Fitting \ Test}$', fontsize=18)

    fig.savefig('mock_data/plots/star_fit.svg', dpi=120, bbox_inches='tight')



def test_psf_coeff_fit():
    # Set up the PSF
    psf_size = 63
    psf_sigma = 3.
    psf_coeffs = np.zeros((6,psf_size,psf_size), dtype='f8')
    psf_coeffs[0] = np.array(astropy.convolution.Gaussian2DKernel(
        psf_sigma, x_size=psf_size, y_size=psf_size,
        mode='oversample', factor=5
    ))
    psf_coeffs[1] = 0.1 * np.array(astropy.convolution.Gaussian2DKernel(
        0.75*psf_sigma, x_size=psf_size, y_size=psf_size,
        mode='oversample', factor=5
    ))
    psf_coeffs[5] = -0.1 * np.array(astropy.convolution.Gaussian2DKernel(
        0.5*psf_sigma, x_size=psf_size, y_size=psf_size,
        mode='oversample', factor=5
    ))
    psf_coeffs[3] = -0.5 * (
        np.array(astropy.convolution.Gaussian2DKernel(
            1.3*psf_sigma, x_size=psf_size, y_size=psf_size,
            mode='oversample', factor=5
            ))
      - 0.9 * np.array(astropy.convolution.Gaussian2DKernel(
            1.1*psf_sigma, x_size=psf_size, y_size=psf_size,
            mode='oversample', factor=5
        ))
    )
    psf_coeffs = normalize_psf_coeffs(psf_coeffs)

    # Generate the stellar postage stamps
    n_stars = 300
    (ps_img, ps_weight, ps_mask,
     x_star, y_star,
     flux_model, sky_model,
     ccd_shape) = gen_mock_postage_stamps(psf_coeffs, n_stars,
                                          flux_max=1.e4,
                                          mask_prob=0.25,
                                          max_mask_size=0.25)

    # Need to convert the fluxes into "improper" fluxes (i.e., not a true flux,
    # but a multiple of the local PSF, which is not necessarily normalized)
    psf_flux = calc_psf_fluxes(psf_coeffs, x_star, y_star, ccd_shape)

    # Fit the PSF coefficients, putting in the correct flux and sky levels
    psf_coeffs_fit = fit_psf_coeffs(
        flux_model / psf_flux,
        sky_model,
        x_star,
        y_star,
        ccd_shape,
        ps_img,
        ps_weight,
        ps_mask,
        sigma_nonzero_order=1.
    )
    psf_coeffs_fit = normalize_psf_coeffs(psf_coeffs_fit)

    # Plot the model and fitted PSF coefficients
    vmax = np.max([np.max(np.abs(psf_coeffs)), np.max(np.abs(psf_coeffs_fit))])

    fig = plt.figure(figsize=(12,6), dpi=120)
    l = 0
    for j,coeffs in enumerate([psf_coeffs, psf_coeffs_fit, psf_coeffs_fit-psf_coeffs]):
        for k,img in enumerate(coeffs):
            l += 1
            ax = fig.add_subplot(3, 6, l)
            ax.imshow(img.T, origin='upper', aspect='equal', interpolation='nearest',
                             cmap='bwr_r', vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

            if k == 0:
                txt = ['Truth', 'Fit', 'Residual'][j]
                ax.set_ylabel(r'$\mathrm{{ {} }}$'.format(txt), fontsize=18)

    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    fig.suptitle(r'$\mathrm{PSF \ Fitting \ Test}$', fontsize=18)
    fig.savefig('mock_data/plots/psf_coeff_fit.svg', bbox_inches='tight')


def test_shift_stars():
    max_shift = 5

    # Set up the PSF
    psf_size = 63 + 2 * max_shift
    psf_sigma = 3.
    psf_coeffs = np.zeros((6,psf_size,psf_size), dtype='f8')
    psf_coeffs[0] = np.array(astropy.convolution.Gaussian2DKernel(
        psf_sigma, x_size=psf_size, y_size=psf_size,
        mode='oversample', factor=5
    ))
    psf_coeffs[3] = 0.1 * np.array(astropy.convolution.Gaussian2DKernel(
        0.5*psf_sigma, x_size=psf_size, y_size=psf_size,
        mode='oversample', factor=5
    ))
    psf_coeffs = normalize_psf_coeffs(psf_coeffs)

    # Generate the stellar postage stamps
    n_stars = 1000
    (ps_img, ps_weight, ps_mask,
     x_star, y_star,
     flux_model, sky_model,
     ccd_shape) = gen_mock_postage_stamps(psf_coeffs, n_stars,
                                          flux_max=8.e2,
                                          mask_prob=0.25,
                                          max_mask_size=0.1)

    # Shift each star
    dx_true = max_shift * (2. * np.random.random(n_stars) - 1.)
    dy_true = max_shift * (2. * np.random.random(n_stars) - 1.)

    for k in range(n_stars):
        ps_img[k] = sinc_shift_image(ps_img[k], dx_true[k], dy_true[k])
        ps_weight[k] = sinc_shift_image(ps_weight[k], dx_true[k], dy_true[k])
        # Shift the mask by the nearest integer amount
        ps_mask[k] = np.roll(np.roll(ps_mask[k], int(np.around(dx_true[k])), axis=0),
                             int(np.around(dy_true[k])), axis=1)

    # Clip the postage stamps
    ps_img = ps_img[:, max_shift:-max_shift, max_shift:-max_shift]
    ps_weight = ps_weight[:, max_shift:-max_shift, max_shift:-max_shift]
    ps_mask = ps_mask[:, max_shift:-max_shift, max_shift:-max_shift]
    psf_coeffs = psf_coeffs[:, max_shift:-max_shift, max_shift:-max_shift]

    # Apply the mask
    idx = (ps_mask > 1.e-5)
    ps_img[idx] = np.nan
    ps_weight[idx] = 0.

    # Expand the mask by one pixel
    kern = astropy.convolution.Box2DKernel(3)
    for k in range(n_stars):
        ps_mask[k] = astropy.convolution.convolve(ps_mask[k], kern, boundary='extend')

    ps_mask[(ps_mask < 0.1)] = 0.

    # Fit stellar offset
    dx_fit = np.empty(n_stars, dtype='f8')
    dy_fit = np.empty(n_stars, dtype='f8')

    #kern = astropy.convolution.Gaussian2DKernel(3.)

    for k in range(n_stars):
        #ps_img[k] = astropy.convolution.convolve(ps_img[k], kern, boundary='extend')
        dx_fit[k], dy_fit[k] = fit_star_offset(
            psf_coeffs,
            ps_img[k],
            ps_weight[k],
            ps_mask[k],
            x_star[k],
            y_star[k],
            flux_model[k],
            sky_model[k],
            ccd_shape,
            max_shift=2.*max_shift,
            verbose=True
        )

    # Distance from true position
    ds = np.sqrt((dx_fit - dx_true)**2. + (dy_fit - dy_true)**2.)

    # Plot histogram of distance from true position
    fig = plt.figure(figsize=(10,5), dpi=120)

    ax = fig.add_subplot(1,1,1)

    ds_plot = []
    for n0,n1 in zip([0., 100., 200.], [100., 200., np.inf]):
        idx = (flux_model >= n0 * sky_model) & (flux_model < n1 * sky_model)
        ds_plot.append(ds[idx])

    ax.hist(ds_plot, 50, normed=1, histtype='stepfilled', stacked=True)

    ax.set_xlabel(r'$\Delta s \ \left( \mathrm{pixels} \right)$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{Frequency}$', fontsize=16)
    ax.set_title(r'$\mathrm{Distance \ from \ True \ Stellar \ Position}$', fontsize=18)

    fig.savefig('mock_data/plots/shifted_stars_hist.svg', dpi=120, bbox_inches='tight')

    # Plot the shifted stars
    fig = plt.figure(figsize=(12,12), dpi=120)
    i = 0
    for j in range(4):
        for k in range(4):
            ax = fig.add_subplot(4,4,i+1, axisbg='g')

            idx = (ps_mask[i] < 1.e-5)
            vmin, vmax = np.percentile(ps_img[i][idx], [1., 99.])
            ax.imshow(ps_img[i].T, origin='lower', aspect='equal',
                                   interpolation='none', cmap='binary',
                                   vmin=vmin, vmax=vmax)

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            x = [dx_true[i] + 0.5 * (psf_size-2.*max_shift) - 0.5]
            y = [dy_true[i] + 0.5 * (psf_size-2.*max_shift) - 0.5]
            ax.scatter(x, y, edgecolor='none', facecolor='cyan', s=5, alpha=0.5)

            x = [dx_fit[i] + 0.5 * (psf_size-2.*max_shift) - 0.5]
            y = [dy_fit[i] + 0.5 * (psf_size-2.*max_shift) - 0.5]
            ax.scatter(x, y, edgecolor='none', facecolor='r', s=5, alpha=0.5)

            x = xlim[0] + 0.02 * (xlim[1] - xlim[0])
            y = ylim[1] - 0.02 * (ylim[1] - ylim[0])
            txt = r'${:.1f}$'.format(flux_model[i] / sky_model[i])
            ax.text(x, y, txt,
                    fontsize=12, color='b',
                    ha='left', va='top')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            i += 1

    fig.subplots_adjust(hspace=0.02, wspace=0.02)
    fig.suptitle(r'$\mathrm{Star \ Centroiding \ Test}$', fontsize=22, y=0.94)
    fig.savefig('mock_data/plots/shifted_stars.svg', bbox_inches='tight', dpi=120)


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


def gen_test_data(psf_coeffs, sky_level=100., limiting_mag=24., band=3):
    psf_shape = psf_coeffs.shape[1:]

    # Load a test image
    fname_pattern = 'psftest/c4d_150109_051822_oo{}_z_v1.fits.fz'
    img_data, weight_data, mask_data, wcs, exp_num = load_exposure(fname_pattern, 'S31')
    ccd_shape = img_data.shape

    # Load locations of PS1 stars
    fname = 'psftest/ps1stars-c4d_150109_051822.fits'
    ps1_table = astropy.io.fits.getdata(fname, 1)
    star_x, star_y, ps1_mag = get_star_locations(ps1_table, wcs, img_data.shape, min_separation=1.)

    # Transform stellar magnitudes to fluxes
    limiting_mag_flux = 5. * np.sqrt(sky_level)
    star_flux = limiting_mag_flux * 10.**(-0.4 * (ps1_mag[:,band]-limiting_mag))

    # Work on a larger canvas, so that edge effects can be ignored
    img_shape = [ccd_shape[0]+2*psf_shape[0], ccd_shape[1]+2*psf_shape[1]]
    dx = 0.5 * (psf_shape[0] - 1)
    dy = 0.5 * (psf_shape[1] - 1)

    # Fill the image with Gaussian background noise
    img_mean = sky_level * np.ones(shape=img_shape, dtype='f8')
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

    # Poisson statistics
    sigma2 = np.abs(img_mean)
    #sigma2[sigma2 < 0.1] = 0.1    # Floor on variance

    # Add noise into the image
    img_model = img_mean + np.sqrt(sigma2) * np.random.normal(loc=0., scale=1., size=ccd_shape)

    # Calculate the weights (assume Poisson statistics)
    weight_model = 1. / sigma2

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


def test_extract_psf(replace_with_mock=False):
    # Load a test image
    fname_pattern = 'psftest/c4d_150109_051822_oo{}_z_v1.fits.fz'
    #fname_pattern = 'psfunit/detest-00396086-S31-oo{}.fits.fz'
    ccd_id = 'S31'
    img_data, weight_data, mask_data, wcs, exp_num = load_exposure(fname_pattern, ccd_id)
    ccd_shape = img_data.shape

    psf_coeffs_model = psf_coeffs_model = np.zeros((6,63,63), dtype='f8')

    if replace_with_mock:
        psf_sigma = 3.
        psf_coeffs_model[0] = np.array(astropy.convolution.Gaussian2DKernel(
            psf_sigma, x_size=psf_coeffs_model.shape[1], y_size=psf_coeffs_model.shape[2],
            mode='oversample', factor=5
        ))
        #psf_coeffs_model[0] += 0.25 * np.array(astropy.convolution.Gaussian2DKernel(
        #    0.5*psf_sigma, x_size=psf_coeffs_model.shape[1], y_size=psf_coeffs_model.shape[2],
        #    mode='oversample', factor=5
        #))
        #psf_coeffs_model[1] = -0.25 * np.array(astropy.convolution.Gaussian2DKernel(
        #    0.5*psf_sigma, x_size=psf_coeffs_model.shape[1], y_size=psf_coeffs_model.shape[2],
        #    mode='oversample', factor=5
        #))

        img_data, weight_data, mask_data, star_x, star_y  = gen_test_data(
            psf_coeffs_model,
            limiting_mag = 24.
        )
        mask_data[:] = 0

    print '# of masked pixels:', np.sum(mask_data != 0)

    # Extract the PSF
    psf_coeffs, star_dict = extract_psf(img_data, weight_data, mask_data, wcs,
                                        return_postage_stamps=True, n_iter=9,
                                        min_pixel_fraction=0.999999,
                                        star_chisq_threshold=3.,
                                        sky_sigma=1.e9,#0.05,
                                        sigma_nonzero_order=1.)

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
    psf_resid = calc_star_chisq(
        psf_coeffs, star_dict['ps_exposure'],
        star_dict['ps_weight'], star_dict['ps_mask'],
        star_dict['star_x'], star_dict['star_y'],
        star_dict['stellar_flux'], star_dict['sky_level'],
        ccd_shape
    )

    tmp = star_dict['ps_exposure'] * star_dict['ps_weight']
    tmp.shape = (tmp.shape[0], tmp.shape[1]*tmp.shape[2])
    print 'Exposure * weight:'
    print np.median(tmp, axis=1)
    print ''

    # Calculate corrections to fluxes, by normalizing PSFs
    #psf_norm = np.empty(n_stars, dtype='f8')
    #for k in range(n_stars):
    #    # Evaluate the PSF at the location of the star
    #    psf_model = eval_psf(psf_coeffs, star_dict['star_x'][k],
    #                         star_dict['star_y'][k], ccd_shape)
    #    #psf_norm[k] = np.sum(psf_model)

    #star_dict['stellar_flux'] *= psf_norm

    # Scatterplot of PS1 flux with inferred flux
    fig = plt.figure(figsize=(16,8), dpi=100)

    ps1_flux = 10.**(-(star_dict['star_ps1_mag']-20.) / 2.5)
    fit_flux = star_dict['stellar_flux']
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

    # Scatterplot of fit magnitude vs. sky level
    fig = plt.figure(figsize=(8,8), dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.scatter(fit_mag, star_dict['sky_level'],
               edgecolor='none', facecolor='b', s=10)
    ax.set_xlabel(r'$\mathrm{fit \ magnitude}$', fontsize=18)
    ax.set_ylabel(r'$\mathrm{sky \ level}$', fontsize=18)
    ax.set_title(r'$\mathrm{Fit \ Parameters}$', fontsize=20)
    ax.set_xlim(ax.get_xlim()[::-1])

    fig.savefig('star_fit_params.svg', bbox_inches='tight')
    plt.close(fig)

    # Scatterplot of PS1 magnitude vs. fit sky level
    fig = plt.figure(figsize=(8,8), dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.scatter(star_dict['star_ps1_mag'][:,3], star_dict['sky_level'],
               edgecolor='none', facecolor='b', s=10)
    ax.set_xlabel(r'$z_{\mathrm{PS1}}$', fontsize=18)
    ax.set_ylabel(r'$\mathrm{sky \ level}$', fontsize=18)
    ax.set_title(r'$\mathrm{Sky \ Level \ Trend}$', fontsize=20)
    ax.set_xlim([21., 11.])

    fig.savefig('star_fit_sky_level.svg', bbox_inches='tight')
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
    for pc, plt_fname, plt_title in ((psf_coeffs, 'psf_fit_over_ccd', r'$\mathrm{PSF \ Fit}$'),
                                     (psf_coeffs_model, 'psf_model_over_ccd', r'$\mathrm{PSF \ Model}$'),
                                     (psf_coeffs-psf_coeffs_model, 'psf_resid_over_ccd', r'$\mathrm{PSF \ residuals}$')):
        for vmax in [1., 0.1, 0.01, 0.001]:
            fig = plt.figure(figsize=(12,12), dpi=120)

            for j,x in enumerate(np.linspace(0, ccd_shape[0], 3)):
                for k,y in enumerate(np.linspace(0, ccd_shape[1], 3)):
                    tmp = eval_psf(pc, x, y, ccd_shape)
                    tmp /= np.max(np.abs(tmp))

                    ax = fig.add_subplot(3,3,3*k+j+1)

                    ax.imshow(tmp.T, origin='upper', aspect='equal',
                              interpolation='nearest', cmap='bwr_r',
                              vmin=-vmax, vmax=vmax)

                    ax.set_xticks([])
                    ax.set_yticks([])

            fig.subplots_adjust(wspace=0.01, hspace=0.01)

            fig.suptitle(plt_title, fontsize=20)

            fig.savefig('{}_{:.3f}.png'.format(plt_fname, vmax),
                        dpi=120, bbox_inches='tight')
            plt.close(fig)

    '''
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
    '''

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
        tmp[tmp > 0.01] = np.nan

        ax.imshow(tmp.T, origin='upper', aspect='equal',
                  interpolation='nearest', cmap='winter_r',
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
        tmp = 1. / np.sqrt(star_dict['ps_weight'][k])
        tmp[star_dict['ps_mask'][k] != 0] = np.nan

        vmin, vmax = 0., 1.

        idx = np.isfinite(tmp)
        if np.any(idx):
            #vmin, vmax = np.percentile(tmp[idx], [1., 99.8])
            vmin, vmax = np.min(tmp[idx]), np.max(tmp[idx])

        ax = fig.add_subplot(n_x, n_y, k+1, axisbg='g')

        ax.imshow(tmp.T, origin='upper', aspect='equal',
                  interpolation='nearest', cmap='bwr_r',
                  vmin=-vmax, vmax=vmax)

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
        psf_model /= np.sum(psf_model)

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

        txt = r'${:.1f}$'.format(psf_resid[k])
        ax.text(x_txt, y_txt, txt)
        ax_stretch.text(x_txt, y_txt, txt)
        ax_weighted.text(x_txt, y_txt, txt)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_txt = xlim[0] + 0.95 * (xlim[1] - xlim[0])
        y_txt = ylim[0] + 0.95 * (ylim[1] - ylim[0])

        txt = r'${:.1f}$'.format(star_dict['star_ps1_mag'][k,3])
        ax.text(x_txt, y_txt, txt, ha='right', va='top')
        ax_stretch.text(x_txt, y_txt, txt, ha='right', va='top')
        ax_weighted.text(x_txt, y_txt, txt, ha='right', va='top')

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
    #test_chisq(add_secondary_sources=False)
    #test_psf_coeff_fit()
    test_extract_psf(replace_with_mock=False)
    #test_gen_data()
    #test_shift_stars()
    #test_star_fit()

    return 0


if __name__ == '__main__':
    main()
