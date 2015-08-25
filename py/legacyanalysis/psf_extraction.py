import numpy as np
import scipy.ndimage

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


def extract_psf(exposure_img, error_img, wcs_header, star_coords):
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


def main():
    test_sinc_shift_image()
    return 0


if __name__ == '__main__':
    main()
