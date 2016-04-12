from __future__ import print_function

import os
import numpy as np

import fitsio

from astrometry.util.fits import fits_table
from astrometry.util.file import trymakedirs
from astrometry.util.util import Tan

from legacypipe.common import LegacySurveyData, wcs_for_brick

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='This script creates small self-contained data sets that '
        'are useful for test cases of the pipeline codes.')

    parser.add_argument('ccds', help='CCDs table describing region to grab')
    parser.add_argument('outdir', help='Output directory name')
    parser.add_argument('brick', help='Brick containing these images')
    
    args = parser.parse_args()

    C = fits_table(args.ccds)
    print(len(C), 'CCDs in', args.ccds)

    survey = LegacySurveyData()
    bricks = survey.get_bricks_readonly()
    outbricks = bricks[np.array([n == args.brick for n in bricks.brickname])]
    assert(len(outbricks) == 1)
    
    outsurvey = LegacySurveyData(survey_dir = args.outdir)
    trymakedirs(args.outdir)
    outbricks.writeto(os.path.join(args.outdir, 'survey-bricks.fits.gz'))

    targetwcs = wcs_for_brick(outbricks[0])
    H,W = targetwcs.shape
    
    tycho = fits_table(os.path.join(survey.get_survey_dir(), 'tycho2.fits.gz'))
    print('Read', len(tycho), 'Tycho-2 stars')
    ok,tx,ty = targetwcs.radec2pixelxy(tycho.ra, tycho.dec)
    margin = 100
    tycho.cut(ok * (tx > -margin) * (tx < W+margin) *
              (ty > -margin) * (ty < H+margin))
    print('Cut to', len(tycho), 'Tycho-2 stars within brick')
    del ok,tx,ty
    tycho.writeto(os.path.join(args.outdir, 'tycho2.fits.gz'))
    
    outccds = C.copy()
    outccds.delete_column('ccd_x0')
    outccds.delete_column('ccd_x1')
    outccds.delete_column('ccd_y0')
    outccds.delete_column('ccd_y1')
    outccds.image_hdu[:] = 1
    
    for iccd,ccd in enumerate(C):

        assert(ccd.camera.strip() == 'decam')

        im = survey.get_image_object(ccd)
        print('Got', im)
        slc = (slice(ccd.ccd_y0, ccd.ccd_y1), slice(ccd.ccd_x0, ccd.ccd_x1))
        tim = im.get_tractor_image(slc, pixPsf=True, splinesky=True,
                                   subsky=False, nanomaggies=False)
        print('Tim:', tim.shape)

        psf = tim.getPsf()
        print('PSF:', psf)
        psfex = psf.psfex
        print('PsfEx:', psfex)

        outim = outsurvey.get_image_object(ccd)
        print('Output image:', outim)

        print('Image filename:', outim.imgfn)
        trymakedirs(outim.imgfn, dir=True)

        # Adjust the header WCS by x0,y0
        crpix1 = tim.hdr['CRPIX1']
        crpix2 = tim.hdr['CRPIX2']
        tim.hdr['CRPIX1'] = crpix1 - ccd.ccd_x0
        tim.hdr['CRPIX2'] = crpix2 - ccd.ccd_y0
        
        fitsio.write(outim.imgfn, None, header=tim.primhdr, clobber=True)
        fitsio.write(outim.imgfn, tim.getImage(), header=tim.hdr,
                     extname=ccd.ccdname)

        h,w = tim.shape
        outccds.width[iccd] = w
        outccds.height[iccd] = h
        outccds.crpix1[iccd] = crpix1 - ccd.ccd_x0
        outccds.crpix2[iccd] = crpix2 - ccd.ccd_y0

        wcs = Tan(*[float(x) for x in
                    [ccd.crval1, ccd.crval2, ccd.crpix1, ccd.crpix2,
                     ccd.cd1_1, ccd.cd1_2, ccd.cd2_1, ccd.cd2_2, ccd.width, ccd.height]])
        subwcs = wcs.get_subimage(ccd.ccd_x0, ccd.ccd_y0, w, h)
        outccds.ra[iccd],outccds.dec[iccd] = subwcs.radec_center()
        
        print('Weight filename:', outim.wtfn)
        trymakedirs(outim.wtfn, dir=True)
        fitsio.write(outim.wtfn, None, header=tim.primhdr, clobber=True)
        fitsio.write(outim.wtfn, tim.getInvvar(), header=tim.hdr,
                     extname=ccd.ccdname)

        print('DQ filename', outim.dqfn)
        trymakedirs(outim.dqfn, dir=True)
        fitsio.write(outim.dqfn, None, header=tim.primhdr, clobber=True)
        fitsio.write(outim.dqfn, tim.dq, header=tim.hdr,
                     extname=ccd.ccdname)
        
        
        print('PSF filename:', outim.psffn)
        trymakedirs(outim.psffn, dir=True)
        psfex.writeto(outim.psffn)

        print('Sky filename:', outim.splineskyfn)
        sky = tim.getSky()
        print('Sky:', sky)
        trymakedirs(outim.splineskyfn, dir=True)
        sky.write_fits(outim.splineskyfn)

    outccds.writeto(os.path.join(args.outdir, 'survey-ccds-1.fits.gz'))

    outC = outsurvey.get_ccds_readonly()
    for iccd,ccd in enumerate(outC):
        outim = outsurvey.get_image_object(ccd)
        print('Got output image:', outim)
        otim = outim.get_tractor_image(pixPsf=True, splinesky=True)
        print('Got output tim:', otim)
    
if __name__ == '__main__':
    main()


