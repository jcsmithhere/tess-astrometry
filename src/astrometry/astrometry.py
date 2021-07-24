""" Tools for performing astrometry analysis on asteroids in TESS data"""

import warnings
from lightkurve.targetpixelfile import TessTargetPixelFile
import lightkurve as lk
from lightkurve.utils import validate_method
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import astropy.units as u
from scipy.stats import median_absolute_deviation
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import label
from astropy.io import fits
from tqdm import tqdm
import csv
from scipy.optimize import minimize_scalar
from tess_ephem import ephem
from scipy.stats import median_abs_deviation
import os
import glob
import pandas as pd
#import multiprocessing as mp


#*************************************************************************************************************
class Centroids:
    """ Object to store computed centroid information

    Atributes
    ---------
    mtpf        : astrometry.MovingTargetPixelFile
        The moving target pixel file to compute centroids for
    time  : astropy.time.core.Time
        Timestamps in BTJD
    aper : ndarray(nCadences, cols, rows)
        The selected aperture per cadence
    col         : [np.array]
        Column centroids in pixels
    row         : [np.array]
        Row centroids in pixels
    ra          : [np.array]
        Right Ascension centroids in degrees
    dec         : [np.array]
        Declination centroids in degrees
    expected_ra : np.array
        The JPL Horizons expected RA
    expected_dec : np.array
        The JPL Horizons expected decl.
    raDec2Pix_ra : np.array
        The SPOC raDec2Pix computed RA
    raDec2Pix_dec : np.array
        The SPOC raDec2Pix computed decl.
    expected_row : np.array
        The JPL Horizons expected row
    expected_col : np.array
        The JPL Horizons expected column
    colRowDataAvailable : bool
        If column and row centroid data in pixels is in object
    raDecDataAvailable : bool
        If RA and Decl. centroid data in degrees is in object
    """

    centroid_type_options = ('colRowData', 'raDecData')

    #*************************************************************************************************************
    def __init__(self, mtpf):
        """ Generate the Centroids object. 
        
        Compute the centrouds using one of the centroiding methods.

        """

        assert isinstance(mtpf, MovingTargetPixelFile), 'Must pass a MovingTargetPixelFile object'

        self.mtpf = mtpf

        self.colRowDataAvailable = False
        self.raDecDataAvailable = False

        self.aper   = None
        self.col    = None
        self.row    = None
        self.ra     = None
        self.dec    = None
    
        # These need to be downloaded with tess-ephem
        self.expected_ra    = None
        self.expected_dec   = None

        # These are computed by the SPOC raDec2Pix:
        self.raDec2Pix_ra    = None
        self.raDec2Pix_dec   = None


    @property
    def time(self):
        """ The time given by the mtpf"""

        return self.mtpf.time

    @property
    def instrument_time(self):
        """ The time given by the mtpf"""

        return self.mtpf.instrument_time

    #*************************************************************************************************************
    def old__init__(self, timestamps, col=None, row=None, ra=None, dec=None):
        """ Creates a Centroids object

        Either col/row or ra/dec can be passed but not both
        """

        if col is not None or row is not None:
            assert col is not None and row is not None, 'Both row and column must be passed or neither'
            assert ra is None and dec is None, 'Must pass either row/col or ra/dec, not both'
            self.colRowData = True
            self.raDecData = False
        elif ra is not None or dec is not None:
            assert ra is not None and dec is not None, 'Both ra and dec must be passed or neither'
            assert col is None and row is None, 'Must pass either row/col or ra/dec, not both'
            self.colRowData = False
            self.raDecData = True
        else:
            raise Exception('Must pass either row/col or ra/dec, and not both')

        self.timestamps = timestamps

        self.row    = np.array(row)
        self.col    = np.array(col)

        self.ra     = np.array(ra)
        self.dec    = np.array(dec)

    #*************************************************************************************************************
    def download_expected_motion(self, aberrate : bool = False):
        """ Downloads the expected motion in R.A. and decl. using tess-ephem, which uses JPL Horizons

        NOTE: By default tess-ephem will perform the approximate DVA correction. This will disable that correction to
        get the true predicted sky coordinates.
        HOWEVER, the RA and Decl. returned by ephem are the true coords, not corrected for R.A. and Decl.

        """

        # Download the JPL Horizons data at the data cadence times
      # print('Downloading Expected R.A. and decl. from JPL Horizons...')
        df = ephem(self.mtpf.targetid, time=self.time, interpolation_step='2m', verbose=True, aberrate=aberrate)

        # pad invalid times with NaN
        # Find missing times
        dfTimeArray = [t.value for t in df.index]
      # missingTimes = np.nonzero(~np.in1d(self.time.value, dfTimeArray))[0]
      # # Fill with Nan
      # df.loc[len(df.index)] = ['Amy', 89, 93] 

        presentTimes = np.nonzero(np.in1d(self.time.value, dfTimeArray))[0]
        self.expected_ra = np.full(len(self.time), np.nan)
        self.expected_ra[presentTimes] = df.ra.values
        self.expected_dec = np.full(len(self.time), np.nan)
        self.expected_dec[presentTimes] = df.dec.values

        self.expected_row = np.full(len(self.time), np.nan)
        self.expected_row[presentTimes] = df.row.values
        self.expected_col = np.full(len(self.time), np.nan)
        self.expected_col[presentTimes] = df.column.values

      # self.expected_ra    = df.ra.values
      # self.expected_dec   = df.dec.values


    #*************************************************************************************************************
    def compute_centroids_simple_aperture(self, method='moments', CCD_ref=True, image_ref=False, aper_mask_threshold=3.0):
        """ Computes the centroid of a moving target pixel file using a simple static aperture
        
        Parameters
        ----------
        method : str
                The centroiding method to use: 'moments' 'quadratic'
        CCD_ref : bool
                If True then add in the mtpf CORNER_COLUMN and CORNER_ROW to get the CCD reference pixel coordinates
        image_ref: bool
                If True then subtract off 0.5 so that the pixel center is at 0.0 
                This aids when superimposing on the pixel grid with matplot lib
                (The TESS convenction is the center is at 0.5)
        aper_mask_threshold : float
            A value for the number of sigma by which a pixel needs to be
            brighter than the median flux to be included in the aperture mask.
        
        Returns
        -------
        centroids : astrometry.Centroids class
            The computed centroids
        
        """
        
        # Calculate the median image
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            median_image = np.array(np.nanmedian(self.mtpf.flux, axis=0))
        vals = median_image[np.isfinite(median_image)].flatten()
        # Calculate the theshold value in flux units
        mad_cut = (1.4826 * median_absolute_deviation(vals) * aper_mask_threshold) + np.nanmedian(median_image)
        # Create a mask containing the pixels above the threshold flux
        aper = np.nan_to_num(median_image) >= mad_cut
        cols, rows = self.mtpf.estimate_centroids(method=method, aperture_mask=aper)
        
        # Subtract off the extra 0.5 so that the centroid is plotted properly on a pixel grid in matplotlib
        if image_ref:
            cols -= 0.5*u.pixel
            rows -= 0.5*u.pixel
        
        if CCD_ref:
            cols += self.mtpf.hdu[1].data['CORNER_COLUMN'][self.mtpf.quality_mask]*u.pixel
            rows += self.mtpf.hdu[1].data['CORNER_ROW'][self.mtpf.quality_mask]*u.pixel
            pass
        
        self.col = cols
        self.row = rows

        self.colRowDataAvailable = True
        
        return

    #*************************************************************************************************************
    def compute_centroids_dynamic_aperture(self, method='moments', CCD_ref=True, image_ref=False,
            aper_mask_threshold=3.0, n_cores=None):
        """ Compute the centroid of a moving target pixel file using a dynamic aperture.
        
            This method will optimize the aperture on each cadence to minimize the sawtooth pattern due to a moving object.

            Computing the centroid on each cadence is an independent process so this function is ripe for asyncronous
            multiprocessing. TODO: figure out how to get this parallel processed given we are using a mtpf class.
        
        Parameters
        ----------
        method : str
                The centroiding method to use: 'moments' 'quadratic'
        CCD_ref : bool
                If True then add in the mtpf CORNER_COLUMN and CORNER_ROW to get the CCD reference pixel coordinates
        image_ref: bool
                If True then subtract off 0.5 so that the pixel center is at 0.0 
                This aids when superimposing on the pixel grid with matplot lib
                (The TESS convenction is the center is at 0.5)
        aper_mask_threshold : float
            A value for the number of sigma by which a pixel needs to be
            brighter than the median flux to be included in the aperture mask.
        n_cores : int
            Number of multiprocessing cores to use. None means use all.
        """

        # Compute the aperture for each cadence seperately
        cols = []
        rows = []
        self.aper = np.full(self.mtpf.shape, np.nan)

      # #***
      # # Multiprocessing
      # pool = mp.Pool(processes=n_cores)
      # # Execute the children in parallel
      # results = [pool.apply_async(_single_cadence_dynamic_aperture, args=(self.mtpf, aper_mask_threshold, method, idx, cadenceno)) for 
      #             idx,cadenceno in enumerate(self.mtpf.cadenceno)]
      # # Collect the results
      # outputs = [result.get() for result in results]
      # for output in outputs:
      #     [aper, col, row] = output
      #     self.aper[idx,:,:] = aper
      #     cols.append(col)
      #     rows.append(row)

      # #***
        

        for idx,cadenceno in enumerate(self.mtpf.cadenceno):
            self.aper[idx,:,:] = self.mtpf.create_threshold_mask_one_cadence(cadenceno, threshold=aper_mask_threshold)
            col, row = self.mtpf.estimate_centroids_one_cadence(cadenceno, method=method,
                    aperture_mask=self.aper[idx,:,:])
            cols.append(col)
            rows.append(row)

        cols = np.array(cols).flatten() * u.pixel
        rows = np.array(rows).flatten() * u.pixel

      # # Subtract off the extra 0.5 so that the centroid is plotted properly on a pixel grid in matplotlib
      # if image_ref:
      #     cols -= 0.5*u.pixel
      #     rows -= 0.5*u.pixel

      # if np.any(np.in1d([1,2], int(np.median(self.mtpf.ccd)))):
      #     # For CCDs 1 and 2
      #     cols += 1.5*u.pixel
      #     rows += 0.5*u.pixel
      # elif np.any(np.in1d([3,4], int(np.median(self.mtpf.ccd)))):
      #     # For CCDs 3 and 4
      #     cols -= 0.5*u.pixel
      #     rows += 0.5*u.pixel
        
        
        if CCD_ref:
            # Fix off-by-one error
            cols += self.mtpf.hdu[1].data['CORNER_COLUMN'][self.mtpf.quality_mask]*u.pixel + 1.0*u.pixel
            rows += self.mtpf.hdu[1].data['CORNER_ROW'][self.mtpf.quality_mask]*u.pixel + 1.0*u.pixel
          # cols += self.mtpf.hdu[1].data['CORNER_COLUMN'][self.mtpf.quality_mask]*u.pixel
          # rows += self.mtpf.hdu[1].data['CORNER_ROW'][self.mtpf.quality_mask]*u.pixel
            pass
        
        self.col = cols
        self.row = rows

        self.colRowDataAvailable = True
        
        return

    
        
    #*************************************************************************************************************
    def write_to_csv(self, data='col_row', filename=None):
        """ Writes out the centroids to a CSV file

        This data is intended for use with the SPOC pipeline tools so the time is converted to instrument time (TJD).

        Parameters
        ----------
        data : str
            What centroid data to write {'col_row', 'ra_dec'}
        filename : str
            Name of file to write to

        """

        assert filename is not None, 'filename must be passed'

        if data == 'col_row':
            assert self.colRowDataAvailable, 'First compute the centroids'
            rows = [[self.mtpf.camera[idx], self.mtpf.ccd[idx], 
                self.instrument_time.value[idx], self.col[idx].value, self.row[idx].value] for idx in np.arange(len(self.row))]
        elif data == 'ra_dec':
            raise Exception('This option is not currently working')
            assert self.raDecDataAvailable, 'First compute the centroids'
            rows = [[self.instrument_time.value[idx], self.ra[idx].value, self.dec[idx].value] for idx in np.arange(len(self.ra))]

        with open(filename, 'w') as fp:
            # Write the header
            fieldnames = ['# target = '+str(self.mtpf.targetid), 
                            ' Sector = '+str(self.mtpf.sector)]
            wr = csv.DictWriter(fp, fieldnames=fieldnames)
            wr.writeheader()

            fieldnames = ['# camera', ' CCD', ' instrument time [TJD]', ' column [pixels]', ' row [pixels]']
            wr = csv.DictWriter(fp, fieldnames=fieldnames)
            wr.writeheader()

            # Write the data
            wr = csv.writer(fp)
            wr.writerows(rows)


    #*************************************************************************************************************
    def read_from_csv(self, data='raDec2Pix_ra_dec', filename=None):
        """ Read data from CSV file.

        Parameters
        ----------
        data : str
            What centroid data to read and store
            Options:
                'raDec2Pix_ra_dec': The ra and dec xomputed by the SPOC raDec2Pix class
        filename : str
            Name of file to read from

        Returns
        -------
        if data='raDec2Pix_ra_dec':
            self.raDec2Pix_ra 
            self.raDec2Pix_dec 
        """

        assert filename is not None, 'filename must be passed'

        self.raDec2Pix_ra    = np.full(len(self.time), np.nan)
        self.raDec2Pix_dec   = np.full(len(self.time), np.nan)

        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                # Find time entry by finding the nearest timestamp to the data
                # TODO: do something more robust
                if data=='raDec2Pix_ra_dec':
                    nearestIdx = np.argmin(np.abs(self.instrument_time.value - float(row[0])))
                    self.raDec2Pix_ra[nearestIdx] = float(row[1])
                    self.raDec2Pix_dec[nearestIdx] = float(row[2])
                else:
                    raise Exception ('The only data option is "raDec2Pix_ra_dec"');


    #*************************************************************************************************************
    def detrend_centroids_via_poly (self, polyorderRange=[1,8], sigmaThreshold=5.0, remove_expected=False,
            fig=None, plot=False):
        """ Detrends any trends in the centroid motion via peicewise polynomial fitting.
 
        This function will optionally first remove the JPL Horizona expected centroids if requested.
 
        Then it will identify discontinuities (i.e. orbit boundaries) or any other spurious regions. It will
        then chunk the data around discontinuities and gap the spurious regions. It will then fit the curves to a dynamic order
        polynomial, where the optimal polyorder is chosen based on RMSE.
 
        This function will work on either row/col centroids or ra/dec centroids.
 
        Parameters
        ----------
        polyorderRange : ndarray list(2)
            The upper and lower polyorders to try
        sigmaThreshold : float
            The sigma threshold for finding segments
 
        Returns
        -------
        column, row : `~astropy.units.Quantity`, `~astropy.units.Quantity`
            Floats containing the column and row detrended centroids
 
 
        """
 
        #***
        # First remove the JPL Horizons expected centroids, if requested
        if remove_expected:
            col_exp = self.mtpf.hdu[1].data['TARGET_COLUMN'][self.mtpf.quality_mask]
            row_exp = self.mtpf.hdu[1].data['TARGET_ROW'][self.mtpf.quality_mask]
            self.col -= col_exp
            self.row -= row_exp
 
          # test = remove_expected_trend_elemental(centroids.row, row_exp)
          # test = remove_expected_trend_elemental(centroids.col, col_exp)
 
        #***
        # Peicewise polynomial fit
        if self.colRowDataAvailable:
            detrended_col_centroids, col_polyFitCurve, col_break_point = _detrend_centroids_elemental(self.col, self.mtpf.time,
                    self.mtpf.cadenceno, polyorderRange, sigmaThreshold=sigmaThreshold)
            detrended_row_centroids, row_polyFitCurve, row_break_point = _detrend_centroids_elemental(self.row, self.mtpf.time,
                    self.mtpf.cadenceno, polyorderRange, sigmaThreshold=sigmaThreshold)
            
        elif self.raDecData:
            detrended_ra_centroids, ra_polyFitCurve, ra_break_point = _detrend_centroids_elemental(centroids.ra, mtpf.time,
                    mtpf.cadenceno, polyorderRange, sigmaThreshold=sigmaThreshold)
            detrended_dec_centroids, dec_polyFitCurve, dec_break_point = _detrend_centroids_elemental(centroids.dec, mtpf.time,
                    mtpf.cadenceno, polyorderRange, sigmaThreshold=sigmaThreshold)
            
 
        # Plot the detrended centroids
        # We can plot either row/col or ra/dec
        if plot:
            if fig is None:
                fig,ax = plt.subplots(1,1, figsize=(12, 10))
            else:
                fig.clf()
 
            #***
            # Raw centroids
            ax = plt.subplot(3,1,1)
            if self.colRowDataAvailable:
                ax.plot(self.mtpf.time.value, self.col, '*b', label='Column Centroids')
                ax.plot(self.mtpf.time.value[col_break_point], self.col[col_break_point], '*r', markersize=10, label='Column Breakpoints')
                ax.plot(self.mtpf.time.value, col_polyFitCurve, '-m', label='Column PolyFit')
                ax.plot(self.mtpf.time.value, self.row, '*c', label='Row Centroids')
                ax.plot(self.mtpf.time.value[row_break_point], self.row[row_break_point], '*r', markersize=10, label='Row Breakpoints')
                ax.plot(self.mtpf.time.value, row_polyFitCurve, '-m', label='Row PolyFit')
                minVal = np.nanmin([self.col, self.row])
                maxVal = np.nanmax([self.col, self.row])
            elif centroids.raDecData:
                ax.plot(mtpf.time.value, centroids.ra, '*b', label='R.A. Centroids')
                ax.plot(mtpf.time.value[ra_break_point], centroids.ra[ra_break_point], '*r', markersize=10, label='R.A. Breakpoints')
                ax.plot(mtpf.time.value, ra_polyFitCurve, '-m', label='R.A. PolyFit')
                ax.plot(mtpf.time.value, centroids.dec, '*c', label='Decl. Centroids')
                ax.plot(mtpf.time.value[dec_break_point], centroids.dec[dec_break_point], '*r', markersize=10, label='Decl. Breakpoints')
                ax.plot(mtpf.time.value, dec_polyFitCurve, '-m', label='Decl. PolyFit')
                minVal = np.nanmin([centroids.ra, centroids.dec])
                maxVal = np.nanmax([centroids.ra, centroids.dec])
            # Plot the momentum dump locations
            # momentum dump bit = 32
            dumpHere = np.nonzero(self.mtpf.hdu[1].data['QUALITY'] & 32 > 0)[0]
           #for idx in dumpHere:
           #    ax.plot([mtpf.hdu[1].data['TIME'][dumpHere], mtpf.hdu[1].data['TIME'][dumpHere]], [minVal, maxVal], '-k')
            ax.vlines(self.mtpf.hdu[1].data['TIME'][dumpHere], ymin=minVal, ymax=maxVal, colors='k', label='Momentum Dumps')
            ax.legend()
            ax.set_title('Removing long term trends in centroids')
            ax.grid()
 
            #***
            # Residual from polynomial
            ax = plt.subplot(3,1,2)
            if self.colRowDataAvailable:
                col = detrended_col_centroids
                row = detrended_row_centroids
                madstd = lambda x: 1.4826*median_absolute_deviation(x, nan_policy='omit')
                ax.plot(self.mtpf.time.value, col, '*-b', label='Column Residual; madstd={:.3f}'.format(madstd(col)))
                ax.plot(self.mtpf.time.value, row, '*-c', label='Row Residual; madstd={:.3f}'.format(madstd(row)))
                ax.set_ylabel('Pixels')
                minVal = np.nanmin([col, row])
                maxVal = np.nanmax([col, row])
            elif centroids.raDecData:
                # plot ra and dec in arcseconds
                ra  = detrended_centroids.ra * 60**2
                dec = detrended_centroids.dec * 60**2
                ax.plot(mtpf.time.value, ra, '*-b', label='R.A. Residual; std={:.3f}'.format(np.nanstd(ra)))
                ax.plot(mtpf.time.value, dec, '*-c', label='Decl. Residual; std={:.3f}'.format(np.nanstd(dec)))
                ax.set_ylabel('Arcseconds')
                minVal = np.nanmin([ra, dec])
                maxVal = np.nanmax([ra, dec])
           #for idx in dumpHere:
           #    ax.plot([mtpf.hdu[1].data['TIME'][dumpHere], mtpf.hdu[1].data['TIME'][dumpHere]], [minVal, maxVal], '-k')
            ax.vlines(self.mtpf.hdu[1].data['TIME'][dumpHere], ymin=minVal, ymax=maxVal, colors='k', label='Momentum Dumps')
            # Plot where the pixel grid snaps a pixel
            ax.legend()
            ax.set_title('Centroid Residuals')
            ax.grid()
 
            #***
            # Examine the Periodogram
            # Convert centroids to LightCurve objects
            ax = plt.subplot(3,1,3)
            if self.colRowDataAvailable:
                col_lc = lk.LightCurve(time=self.mtpf.time, flux=detrended_col_centroids)
                row_lc = lk.LightCurve(time=self.mtpf.time, flux=detrended_row_centroids)
                col_pg = col_lc.to_periodogram()
                row_pg = row_lc.to_periodogram()
                col_pg.plot(ax=ax, view='period', scale='log', label='Column', c='b')
                row_pg.plot(ax=ax, view='period', scale='log', label='Row', c='c')
            elif centroids.raDecData:
                ra_lc   = lk.LightCurve(time=mtpf.time, flux=detrended_centroids.ra)
                dec_lc  = lk.LightCurve(time=mtpf.time, flux=detrended_centroids.dec)
                ra_pg   = ra_lc.to_periodogram()
                dec_pg  = dec_lc.to_periodogram()
                ra_pg.plot(ax=ax, view='period', scale='log', label='R.A.', c='b')
                dec_pg.plot(ax=ax, view='period', scale='log', label='Decl.', c='c')
 
            ax.grid()
            ax.set_title('Periodram of Residual Motion')
            
 
 
        return detrended_col_centroids, detrended_row_centroids
    

    #*************************************************************************************************************
    def detrend_centroids_expected_trend(self, include_DVA=True, extra_title="", plot=False):
        """ Detrends the centroids using the given expected trends

        Parameters
        ----------
        include_DVA : bool
            If True then use includes the DVA term when computing the expected trend
        """
 
        # The JPL Horizons expected locations
      # col_exp = self.mtpf.hdu[1].data['TARGET_COLUMN'][self.mtpf.quality_mask]
      # row_exp = self.mtpf.hdu[1].data['TARGET_ROW'][self.mtpf.quality_mask]
        self.download_expected_motion(aberrate=include_DVA)

 
        rowExpRemoved = _remove_expected_trend_elemental(self.row, self.expected_row)
        colExpRemoved = _remove_expected_trend_elemental(self.col, self.expected_col)
 
        if plot:

            # Now print the results
            fig,ax = plt.subplots(1,1, figsize=(12, 10))
        
            # Initial and expected centroids
            ax = plt.subplot(2,1,1)
            ax.plot(self.mtpf.time.value, self.col, '*b', label='Column Centroids')
            ax.plot(self.mtpf.time.value, self.expected_col, '-m', label='Column Expected')
            ax.plot(self.mtpf.time.value, self.row, '*c', label='Row Centroids')
            ax.plot(self.mtpf.time.value, self.expected_row, '-m', label='Row Expected')
            plt.legend()
            plt.title(extra_title + 'Measured Astrometry vs. JPL Horizons Expected')
            plt.grid()
        
          # # Removing the moded motion
          # ax = plt.subplot(4,1,2)
          # ax.plot(mtpf.time.value, colExpRemoved, '*b', label='Column Centroids')
          # ax.plot(mtpf.time.value, colFittedTrend + np.nanmean(colExpRemoved), '-m', label='Column Fitted Modded Trend')
          # ax.plot(mtpf.time.value, rowExpRemoved, '*c', label='Row Centroids')
          # ax.plot(mtpf.time.value, rowFittedTrend + np.nanmean(rowExpRemoved), '-m', label='Row Fitted Moded Trend')
          # plt.legend()
          # plt.title('With Expected Removed')
          # plt.grid()
        
            # Final Residual
            ax = plt.subplot(2,1,2)
            madstd = lambda x: 1.4826*median_absolute_deviation(x, nan_policy='omit')
            ax.plot(self.mtpf.time.value, colExpRemoved, '*b', label='Column Residual; madstd={:.3f}'.format(madstd(colExpRemoved)))
            ax.plot(self.mtpf.time.value, rowExpRemoved, '*c', label='Row Residual; madstd={:.3f}'.format(madstd(colExpRemoved)))
            # Set plot limits to ignore excursions
            allYData = np.concatenate((colExpRemoved, rowExpRemoved))
          # yUpper = np.nanpercentile(allYData, 99.5)
          # yLower = np.nanpercentile(allYData, 0.5)
            yUpper = np.nanpercentile(allYData, 99.0)
            yLower = np.nanpercentile(allYData, 1.0)
            ax.set_ylim(yLower, yUpper)
            ax.set_ylim(-1.0, 1.0)
            plt.legend()
            plt.title('Final Residual')
            plt.grid()
        
          # # Periodigram
          # ax = plt.subplot(3,1,3)
          # col_lc = lk.LightCurve(time=self.mtpf.time, flux=colExpRemoved)
          # row_lc = lk.LightCurve(time=self.mtpf.time, flux=rowExpRemoved)
          # col_pg = col_lc.to_periodogram()
          # row_pg = row_lc.to_periodogram()
          # col_pg.plot(ax=ax, view='period', scale='log', label='Column', c='b')
          # row_pg.plot(ax=ax, view='period', scale='log', label='Row', c='c')
          # ax.grid()
          # ax.set_title('Periodram of Residual Motion')

        return colExpRemoved, rowExpRemoved

    #*************************************************************************************************************
    def compare_JPL_to_computed_centroids(self, raDec2PixDataPath=None, plot_figure=False, include_DVA=True):
        """ Compares the JPL Horizons expected astrometry to that computed by the centroiding and SPOC's raDec2Pix

        Parameters
        ----------
        raDec2PixDataPath : str
            If a file then load in the raDec2Pix data from this filename
            If a directory then automataically attempts to find thwe correct file based on the Target ID
            If None then checks to see if the data is already loaded, if not then raises an exception
        plot_figure : bool
            If True then display figure
        include_DVA : bool
            If True then use includes the DVA term when computing the expected trend

        Returns
        -------
        summary_stats : pandas.DataFrame
            DataFrame containing the difference stats:
            'ccd'               : int # The median CCD for this target
            'medianCol'         : float # The measured median Column
            'medianRow'         : float # The measured median Row
            'medianExpectedRA'  : float
            'medianExpectedDec' : float
            'medianRaDec2PixRA' : float
            'medianRaDec2PixDec': float
            'stdRaDiff'         : float 
            'stdDecDiff'        : float 
            'medianRaDiff'      : float 
            'medianDecDiff'     : float 
        diff_arrays : dict
            Dictionary containing the astrometry difference time series
            'raDiff'        : float array
            'decDiff'       : float array

        """
        assert self.colRowDataAvailable, 'Must first compute the centroids'
    
        # Load in raDec2Pix computed data is passed
        if os.path.isdir(raDec2PixDataPath):
            # Attempt to automatically find the correct file
            targetid = str(self.mtpf.targetid)
            filenames = glob.glob(os.path.join(raDec2PixDataPath, "*.csv"))
            foundIdx = [idx for idx,file in enumerate(filenames) if '_'+targetid+'_' in file]
            if len(foundIdx) > 1:
                raise Exception('Found more than one filename match')
            else:
                foundIdx = foundIdx[0]
            self.read_from_csv(data='raDec2Pix_ra_dec', filename=filenames[foundIdx])
        if os.path.isfile(raDec2PixDataPath):
            #print('Loading in raDec2Pix computed data...')
            self.read_from_csv(data='raDec2Pix_ra_dec', filename=raDec2PixDataPath)
        elif self.raDec2Pix_ra is None or self.raDec2Pix_dec is None:
            raise Exception('raDec2Pix data is not present, must be loaded.')

        # Load the JPL Horizons expected data if not already loaded
        if self.expected_ra is None or self.expected_dec is None:
            self.download_expected_motion(aberrate=include_DVA)
            

        madstd = lambda x: 1.4826*median_abs_deviation(x, nan_policy='omit')

        # Compute in arcseconds
        medianCol           = np.nanmedian(self.col)
        medianRow           = np.nanmedian(self.row)
        medianExpectedRA    = np.nanmedian(self.expected_ra)
        medianExpectedDec   = np.nanmedian(self.expected_dec)
        medianRaDec2PixRA   = np.nanmedian(self.raDec2Pix_ra)
        medianRaDec2PixDec  = np.nanmedian(self.raDec2Pix_dec)

        raDiff          = (self.raDec2Pix_ra - self.expected_ra) * 60 * 60
        decDiff         = (self.raDec2Pix_dec - self.expected_dec) * 60 * 60
        stdRaDiff       = madstd(raDiff)
        stdDecDiff      = madstd(decDiff)
        medianRaDiff    = np.nanmedian(raDiff)
        medianDecDiff   = np.nanmedian(decDiff)


        if plot_figure:
            fig,ax = plt.subplots(1,1, figsize=(12, 10))
            ax = plt.subplot(3,1,1)
            plt.plot(self.expected_dec, self.expected_ra, '*r', label='JPL Horizons')
            plt.plot(self.raDec2Pix_dec, self.raDec2Pix_ra, '-b', label='Centroiding With SPOC raDec2Pix')
            plt.legend()
            plt.grid()
            plt.title('Target {}; Comparing JPL Horizons predicted astrometry versus measured Centroids and raDec2Pix'.format(self.mtpf.targetid))
            plt.xlabel('Decl. [Deg.]')
            plt.ylabel('R.A. [Deg.]')
            
            # Scale the difference figures so we ignore the excursions and just see the detail
            yLow = np.nanpercentile(decDiff, 3)
            yHigh = np.nanpercentile(decDiff, 97)
            ax = plt.subplot(3,1,2)
            plt.plot(self.time.value, decDiff, '*m', label='Decl. difference; madstd={:.3f}; median={:.3f}'.format(stdDecDiff, medianDecDiff))
            plt.legend()
            plt.grid()
            plt.xlabel('TJD')
            plt.ylabel('Error [Arcsec]')
            plt.ylim(yLow, yHigh)
            
            yLow = np.nanpercentile(raDiff, 3)
            yHigh = np.nanpercentile(raDiff, 97)
            ax=plt.subplot(3,1,3)
            plt.plot(self.time.value, raDiff, '*c', label='R.A. difference; madstd={:.3f}; median={:.3f}'.format(stdRaDiff, medianRaDiff))
            plt.legend()
            plt.grid()
            plt.xlabel('TJD')
            plt.ylabel('Error [Arcsec]')
            plt.ylim(yLow, yHigh)

        summary_stats = pd.DataFrame(
                {
                    'ccd'                   :  int(np.median(self.mtpf.ccd)),
                    'medianCol'             :  medianCol,
                    'medianRow'             :  medianRow,  
                    'medianExpectedRA'      :  medianExpectedRA,
                    'medianExpectedDec'     :  medianExpectedDec,  
                    'medianRaDec2PixRA'     :  medianRaDec2PixRA, 
                    'medianRaDec2PixDec'    :  medianRaDec2PixDec, 
                    'stdRaDiff' :        stdRaDiff,    
                    'stdDecDiff' :       stdDecDiff,   
                    'medianRaDiff' :     medianRaDiff, 
                    'medianDecDiff' :    medianDecDiff
                }, index=[str(self.mtpf.targetid)]
            )

        diff_arrays = {
                'raDiff' :           raDiff,       
                'decDiff' :          decDiff
                }

        return summary_stats, diff_arrays
    

#*************************************************************************************************************
# Define the elemental process for each process in the compute_centroids_dynamic_aperture pool
def _single_cadence_dynamic_aperture (mtpf, aper_mask_threshold, method, idx, cadenceno):
    aper = mtpf.create_threshold_mask_one_cadence(cadenceno, threshold=aper_mask_threshold)
    col, row = mtpf.estimate_centroids_one_cadence(cadenceno, method=method,
            aperture_mask=aper)
    return aper, col, row

#*************************************************************************************************************
def _remove_expected_trend_elemental(centroid, exp_trend, subtract_mean=True):
    """ Removes the expected motion trend from the centroids

    Parameters
    ----------
    subtract_mean : bool
        If True then also subtract the secular so the mean is zero

    Returns
    -------
    detrended_centroid : np.array
        The centroid with both the expected curve and the moded sawtooth removed
    centroid_exp_removed : np.array
        The centroid with just the expected curve removed
    fittedTrend : np.array
        The Fit to the moded trend

    """

    # Remove unit from centroid
    cent = np.array(centroid)


    # First remove the first-order term
    centroid_exp_removed = cent - exp_trend

  # # Remove the secular term in the residual
  # centroid_exp_removed -= np.nanmean(centroid_exp_removed) 

  # # Now remove the moded motion
  # moded_trend = np.mod(exp_trend, 1)
  # moded_trend = (moded_trend / np.mean(moded_trend)) - 1
  # # Perform a simple least-squares fit


  # #***
  # # Use scipy.optimize.minimize_scalar
  # # Minimize the RMSE
  # def rmse(coeff):
  #     return np.sqrt(np.sum((centroid_exp_removed - moded_trend*coeff)**2))

  # minimize_result = minimize_scalar(rmse, method='Bounded',
  #         bounds=[-1,1])
  ##        options={'maxiter':max_iter, 'disp': False})

  # fittedTrend = moded_trend*minimize_result.x
  # detrended_centroid = centroid_exp_removed - fittedTrend 

    return centroid_exp_removed

#*************************************************************************************************************
def _detrend_centroids_elemental(centroids, time, cadenceno, polyorderRange, sigmaThreshold=5.0):
    """ Detrends a single centroids time series.

    Helper function for detrend_centroids_via_poly

    Parameters
    ----------
    centroids : np.array * u.pixel
        The 1 dimensional centroids time series
    time    : astropy.time.core.Time
        Cadences times corresponding to <centroids>
    cadenceno : int list
        Cadence numbers associated with timestamps
    polyorderRange : ndarray list(2)
        The upper and lower polyorders to try
    sigmaThreshold : float
        The sigma threshold for finding segments

    Returns
    -------
    detrended_centroids  : np.array * u.pixel 
    polyFitCurve    : np.array list
        The fitted polynomial curve
    break_point  : int list
        The cadence indices where breaks occur
    
    """

    # Remove unit from centroids
    cent = np.array(centroids)

    # Use first differences to find abrupt changes in the centroids
    # First fill gaps using PCHIP interpolation
    # We need to fill timestamps too
    filled_cadenceno = np.arange(cadenceno[0], cadenceno[-1]+1)
    # Fill the centroids to the filled timstamps
    fInterpPCHIP = PchipInterpolator(cadenceno, cent, extrapolate=False)
    centroids_interpolated = fInterpPCHIP(filled_cadenceno)
    
    diffs = np.diff(centroids_interpolated)
    # Only keep diffs associated with actual cadences
    diffs = diffs[np.nonzero( np.in1d(filled_cadenceno[1:], cadenceno) )[0]]

    # Use a median absolute deviation derived standard deviation to find the threshold for outliers
    # sigma = 1.4826 * mad(x)
    # Median normalize then find outliers
    diffsNorm = (diffs / np.median(diffs)) - 1
    colSigma = 1.4826*median_absolute_deviation(diffsNorm)
    threshold = colSigma*sigmaThreshold
    # We want the cadence after the big difference jump (so add 1)
    break_point = np.nonzero(np.abs(diffsNorm) >= threshold)[0] + 1

    # Remove a polynomial to each chunk.
    centroidssDetrended = np.full(np.shape(cent), np.nan)
    polyFitCurve        = np.full(np.shape(cent), np.nan)
    centroidsDetrended  = np.full(np.shape(cent), np.nan)
    for iChunk in np.arange(len(break_point)+1):
        if iChunk == 0:
            chunkStart = 0
        else:
            chunkStart = break_point[iChunk-1]

        if iChunk == len(break_point):
            chunkEnd = len(diffs)
        else:
            chunkEnd = np.min(break_point[iChunk], 0)

        if chunkEnd - chunkStart < 2:
            # break points right next to each other, Just fill in with original value
            polyFitCurve[chunkStart] = cent[chunkStart]
        else:
            polyFitCurve[chunkStart:chunkEnd], _ = _find_best_polyfit(time[chunkStart:chunkEnd].value, cent[chunkStart:chunkEnd], polyorderRange)


    detrended_centroids = (cent - polyFitCurve) * u.pixel

    return detrended_centroids, polyFitCurve, break_point

#*************************************************************************************************************
def _find_best_polyfit(x, y, polyorderRange):
    """ Finds the best polyorder to minimise RMS with the <polyOrderRange>

    Helper function for detrend_centroids 

    Parameters
    ----------
    x   : float ndarray
    y   : float ndarray
    polyorderRange : ndarray list(2)
        The upper and lower polyorders to try

    Returns
    -------
    coeffs  : ndarray list
        The returned polynomial coefficients
    order   : int
        The optimal polyorder
    """

    def rmse(y, yFit):
        return np.sqrt(np.sum((y - yFit)**2))


    best_polyorder = 0
    best_rmse = np.inf
    for polyorder in np.arange(polyorderRange[0], polyorderRange[1]+1):
        pSeries = np.polynomial.polynomial.Polynomial.fit(x, y, polyorder)
        coeffs = pSeries.convert().coef
        polyFitCurve = np.polynomial.polynomial.polyval(x, coeffs)

        if rmse(y, polyFitCurve) < best_rmse:
            best_rmse = rmse(y, polyFitCurve)
            best_polyorder = polyorder
            best_polyFitCurve = polyFitCurve

    return best_polyFitCurve, best_polyorder
    

#*************************************************************************************************************
def plot_centroids_2D (centroids):
    """ Plots the centroids in 2D pixel space
    """

    plt.plot(centroids.col, centroids.row, '-*b')

    return

#*************************************************************************************************************
def plot_peaks_in_centroid_motion(fits_files, maximum_period=2):
    """ Scatter plots the peaks in periodgrams of the residual centroid motion.

    Compares to the peaks in the light curve.

    Parameters
    ----------
    fits_files : list of str
        The list of FITS files to extract detrended centroids from
    """
    
    centroid_peaks_row = np.full(len(fits_files), np.nan)
    centroid_peaks_col = np.full(len(fits_files), np.nan)
    lc_peaks = np.full(len(fits_files), np.nan)
    avg_col_motion = np.full(len(fits_files), np.nan)
    avg_row_motion = np.full(len(fits_files), np.nan)

    # Compute the centroid and light curve peaks
    for idx in tqdm(range(len(fits_files)), 'Computing centroid and light curve peaks.'):
        f = fits.open(fits_files[idx])
        mtpf = MovingTargetPixelFile(f, quality_bitmask="hardest")
        centroids_abs = compute_centroids(mtpf, CCD_ref=True)
        detrended_centroids = detrend_centroids(mtpf, centroids_abs, polyorderRange=[1,8], debug=False)

        col_peak_days, row_peak_days  = find_peaks_in_centroid_motion(mtpf, detrended_centroids)
        centroid_peaks_col[idx], centroid_peaks_row[idx]  = col_peak_days.value, row_peak_days.value

        # Compute the light curve peaks
        lc = mtpf.to_lightcurve().remove_outliers()
        lc_pg = lc.to_periodogram(maximum_period=maximum_period)
        lc_peaks[idx] = lc_pg.period_at_max_power.value

        # Compute proper motion of centroids
        avg_col_motion[idx], avg_row_motion[idx] = compute_centroid_proper_motion(centroids_abs,  mtpf.time)


    # Plot centroid peak power col vs row
    fig,ax = plt.subplots(1,1)
    plt.plot(centroid_peaks_col, centroid_peaks_row, '*')
    plt.plot([0.0,maximum_period],[0.0, maximum_period], '-k')
    plt.grid()
    plt.title('Period Peak in Centroid Motion Residual')
    plt.xlabel('Column Peak Power Period (days)')
    plt.ylabel('Row Peak Power Period (days)')
    plt.show()

    #***
    # plot centroids power peak vs light curve power peak
    fig,ax = plt.subplots(2,1)
    ax = plt.subplot(2,1,1)
    ax.plot(lc_peaks, centroid_peaks_col, '*')
    ax.plot([0.0,maximum_period],[0.0, maximum_period], '-k')
    ax.grid()
    ax.set_title('Period Peak in lightcurve vs. centroids')
    plt.xlabel('LC Peak Period (days)')
    plt.ylabel('Column Peak Power Period (days)')

    ax = plt.subplot(2,1,2)
    ax.plot(lc_peaks, centroid_peaks_row, '*')
    ax.plot([0.0,maximum_period],[0.0, maximum_period], '-k')
    ax.grid()
    ax.set_title('Period Peak in lightcurve vs. centroids')
    plt.xlabel('LC Peak Period (days)')
    plt.ylabel('Row Peak Power Period (days)')

    plt.tight_layout(pad=0.7)
    plt.show()

    #***
    # Plot centroids power peak versus proper motion
    fig,ax = plt.subplots(2,1)
    ax = plt.subplot(2,1,1)
    ax.plot(avg_col_motion, centroid_peaks_col, '*')
    ax.grid()
    ax.set_title('Column Proper motion vs. centroids')
    plt.xlabel('Column Proper Motion (pixels / day)')
    plt.ylabel('Column Peak Power Period (days)')

    ax = plt.subplot(2,1,2)
    ax.plot(avg_row_motion, centroid_peaks_row, '*')
    ax.grid()
    ax.set_title('Row Proper motion vs. centroids')
    plt.xlabel('Row Proper Motion (pixels / day)')
    plt.ylabel('Row Peak Power Period (days)')

    plt.tight_layout(pad=0.7)
    plt.show()

    return


#*************************************************************************************************************
def find_peaks_in_centroid_motion(mtpf, centroids, maximum_period=2):
    """ Finds the peak in the periodogram for the centroids

    Parameters
    ----------
    mtpf : lightkurve.targetpixelfile.MovingTargetPixelFile
            The moving target pixel file to use
    centroids : list of astrometry.Centroids class
    maximum_period : float
        The maximum perido in days to consider (I.e. to ignore long term trends)

    Returns
    -------
    col_peak_period : float*u.day
        The column centroid preiod of peak power
    row_peak_period : float*u.day
        The row centroid preiod of peak power
    """

    col_lc = lk.LightCurve(time=mtpf.time, flux=centroids.col)
    row_lc = lk.LightCurve(time=mtpf.time, flux=centroids.row)
    col_pg = col_lc.to_periodogram(maximum_period=maximum_period)
    row_pg = row_lc.to_periodogram(maximum_period=maximum_period)

    col_peak_period = col_pg.period_at_max_power
    row_peak_period = row_pg.period_at_max_power

    return col_peak_period, row_peak_period

#*************************************************************************************************************
def compute_centroid_proper_motion(centroids, time):
    """ This will compute the proper motion of the tartet centroids.

    It simply computes the average motion as distance per time in units of pixels per day

    Parameters
    ----------
    centroids : np.array list
        The 1 dimensional centroids time series
    time    : astropy.time.core.Time
        Cadences times corresponding to <centroids>

    Returns
    -------
    avg_col_motion : float
        pixels per day
    avg_row_motion : float
        pixels per day
    """

    # Column proper motion
    colDiff = np.diff(centroids.col)
    timeDiff = np.diff(time.value)
    avg_col_motion = np.abs(np.nanmedian(colDiff / timeDiff))

    # Row proper motion
    rowDiff = np.diff(centroids.row)
    avg_row_motion = np.abs(np.nanmedian(rowDiff / timeDiff))

    return avg_col_motion, avg_row_motion


#*************************************************************************************************************

class MovingTargetPixelFile(TessTargetPixelFile):
    """ This is a modified TessTargetPixelFile for custom methods for moving asteroid astrometry

    """

    
    def __init__(self, path, quality_bitmask="default", **kwargs):
        """ See the TargetPixelFile __init__ for arguments
        """

        # Call the TargetPixelFile Constructor
        super(MovingTargetPixelFile, self).__init__(path, quality_bitmask=quality_bitmask, **kwargs)

        # The target ID 
        self.targetid = path[0].header['TARGET']

    @property
    def camera(self):
        """ The camera is an array """
        return self.hdu[1].data['CAMERA'][self.quality_mask]

    @property
    def ccd(self):
        """ The ccd is an array """
        return self.hdu[1].data['CCD'][self.quality_mask]

    def __repr__(self):
        return "MovingTargetPixelFile(Target ID: {})".format(self.targetid)

    @property
    def instrument_time(self):
        """ The target pixel files are barycenter corrected. This method will remove that correction using 
        the TIMECORR column in the FITs file.
        """

        return self.time - self.hdu[1].data['TIMECORR'][self.quality_mask]


    def animate(self, step: int = None, interval: int = 200, save_file=None, fps=30, **plot_args):
        """Displays an interactive HTML matplotlib animation.

        This feature requires a Jupyter notebook environment to display correctly.

        Parameters
        ----------
        step : int
            Spacing between frames.  By default, the spacing will be determined such that
            50 frames are shown, i.e. `step = len(tpf) // 50`.  Showing more than 50 frames
            will be slow on many systems.
        interval : int
            Delay between frames in milliseconds.
        save_file : str
            Name of file to save, None to not save file
        **plot_args : dict
            Optional parameters passed to tpf.plot().
        """
        try:
            # To make installing Lightkurve easier, ipython is an optional dependency,
            # because we can assume it is installed when notebook-specific features are called
            from IPython.display import HTML
            return HTML(self._to_matplotlib_animation(step=step, interval=interval, 
                save_file=save_file, fps=fps, **plot_args).to_jshtml())
        except ModuleNotFoundError:
            log.error("ipython needs to be installed for animate() to work (e.g., `pip install ipython`)")


    def _to_matplotlib_animation(
        self, step: int = None, interval: int = 200, centroid=None, aperture_mask=None, save_file=None, fps=30, **plot_args
    ) -> "matplotlib.animation.FuncAnimation":
        """Returns a `matplotlib.animation.FuncAnimation` object.

        The animation shows the flux values over time by calling `tpf.plot()` for multiple frames.

        Parameters
        ----------
        step : int
            Spacing between frames.  By default, the spacing will be determined such that
            50 frames are shown, i.e. `step = len(tpf) // 50`.  Showing more than 50 frames
            will be slow on many systems.
        interval : int
            Delay between frames in milliseconds.
        column : str
            Column in TPF to plot
        centroid : float np.array(nCadences,2)
            Centroid data to plot overlaid on the pixel data
            Be sure to plot relative pixels (CCD_ref=False)
        aperture_mask : ndarray(nCadences, cols, rows)
            Highlight pixels selected by aperture_mask.
        save_file : str
            Name of file to save, None to not save file
        fps : int
            Frames per second when saving file
        **plot_args : dict
            Optional parameters passed to tpf.plot().
        """
        if step is None:
            step = len(self) // 50
            if step < 1:
                step = 1

        column = plot_args.get("column", "FLUX")
        ax = self.plot(**plot_args)
        if centroid is not None:
            sct = ax.scatter(centroid[step, 0], centroid[step, 1], c='k')

        def add_aper(aperture_mask_single, ax):
            mask_color="red"
            aperture_mask_single = self._parse_aperture_mask(aperture_mask_single)
            # Remove any current patches to reset the aperture mask in the animation
            if ax.patches != []:
                ax.patches = []
            for i in range(self.shape[1]):
                for j in range(self.shape[2]):
                    if aperture_mask_single[i, j]:
                        rect = patches.Rectangle(
                            xy=(j + self.column - 0.5, i + self.row - 0.5),
                            width=1,
                            height=1,
                            color=mask_color,
                            fill=False,
                            hatch="//",
                        )
                        ax.add_patch(rect)

    
        if aperture_mask is not None:
            add_aper(aperture_mask[step,:,:], ax)


        def init():
            return ax.images

        def animate(i):
            frame = i * step
            ax.images[0].set_data(self.hdu[1].data[column][self.quality_mask][frame])
            # Rescale the color range for each frame
            ax.images[0].set_clim(np.min(self.hdu[1].data[column][self.quality_mask][frame]), 
                    np.max(self.hdu[1].data[column][self.quality_mask][frame]))
            if centroid is not None:
                sct.set_offsets(np.array([centroid[frame, 0], centroid[frame, 1]]))
            if aperture_mask is not None:
                add_aper(aperture_mask[frame,:,:], ax)
            ax.set_title(f"Frame {frame}")
           #return [ax.images, sct]
            return

        plt.close(ax.figure)  # prevent figure from showing up in interactive mode

        # `blit=True` means only re-draw the parts that have changed.
        frames = len(self) // step
        anim = matplotlib.animation.FuncAnimation(
            ax.figure,
            animate,
            init_func=init,
            frames=frames,
            interval=interval,
            blit=False,
        )

        if (save_file is not None):
            anim.save(save_file, writer='imagemagick', fps=fps)

        return anim

    def estimate_centroids_one_cadence(self, cadenceno, aperture_mask="default", method="moments"):
        """ Computes the centroid for a single cadence

        See estimate_centroids for details of methods used.

        Parameters
        ----------
        cadenceno : int
            Cadence index to compute centroid for
        aperture_mask : 'pipeline', 'threshold', 'all', 'default', or array-like
            Which pixels contain the object to be measured, i.e. which pixels
            should be used in the estimation?  If None or 'all' are passed,
            all pixels in the pixel file will be used.
            If 'pipeline' is passed, the mask suggested by the official pipeline
            will be returned.
            If 'threshold' is passed, all pixels brighter than 3-sigma above
            the median flux will be used.
            If 'default' is passed, 'pipeline' mask will be used when available,
            with 'threshold' as the fallback.
            Alternatively, users can pass a boolean array describing the
            aperture mask such that `True` means that the pixel will be used.
        method : 'moments' or 'quadratic'
            Defines which method to use to estimate the centroids. 'moments'
            computes the centroid based on the sample moments of the data.
            'quadratic' fits a 2D polynomial to the data and returns the
            coordinate of the peak of that polynomial.

        Returns
        -------
        column, row : `~astropy.units.Quantity`, `~astropy.units.Quantity`
            Floats containing the column and row positions for the centroid
            on the specified cadence, or NaN for where the estimation failed.
        
        """
        method = validate_method(method, ["moments", "quadratic"])
        if method == "moments":
            return self._estimate_centroids_via_moments_one_cadence(cadenceno, aperture_mask=aperture_mask)
        elif method == "quadratic":
            return self._estimate_centroids_via_quadratic_one_cadence(cadenceno, aperture_mask=aperture_mask)

    def _estimate_centroids_via_moments_one_cadence(self, cadenceno, aperture_mask):
        """Compute the "center of mass" of the light based on the 2D moments;
        this is a helper method for `estimate_centroids_one_cadence()`."""
        aperture_mask = self._parse_aperture_mask(aperture_mask)
      # yy, xx = np.indices(self.shape[1:]) + 0.5
        yy, xx = np.indices(self.shape[1:])
        yy = self.row + yy
        xx = self.column + xx
        cadence_idx = np.nonzero(np.in1d(self.cadenceno, cadenceno))[0]
        total_flux = np.nansum(self.flux[cadence_idx, aperture_mask])
        with warnings.catch_warnings():
            # RuntimeWarnings may occur below if total_flux contains zeros
            warnings.simplefilter("ignore", RuntimeWarning)
            col_centr = (
                    np.nansum(xx * aperture_mask * self.flux[cadence_idx, :,:], axis=(1, 2)) / total_flux
            )
            row_centr = (
                np.nansum(yy * aperture_mask * self.flux[cadence_idx, :,:], axis=(1, 2)) / total_flux
            )
        return col_centr * u.pixel, row_centr * u.pixel

    def _estimate_centroids_via_quadratic_one_cadence(self, cadenceno, aperture_mask):
        """Estimate centroids by fitting a 2D quadratic to the brightest pixels;
        this is a helper method for `estimate_centroids_one_cadence()`."""
        aperture_mask = self._parse_aperture_mask(aperture_mask)
        col_centr, row_centr = [], []
        cadence_idx = np.nonzero(np.in1d(self.cadenceno, cadenceno))[0]
        col, row = centroid_quadratic(self.flux[cadence_idx], mask=aperture_mask)
        col_centr.append(col)
        row_centr.append(row)
      # # Finally, we add .5 to the result bellow because the convention is that
      # # pixels are centered at .5, 1.5, 2.5, ...
      # col_centr = np.asfarray(col_centr) + self.column + 0.5
      # row_centr = np.asfarray(row_centr) + self.row + 0.5
        col_centr = np.asfarray(col_centr) + self.column
        row_centr = np.asfarray(row_centr) + self.row
        col_centr = Quantity(col_centr, unit="pixel")
        row_centr = Quantity(row_centr, unit="pixel")
        return col_centr, row_centr

    def create_threshold_mask_one_cadence(self, cadenceno, threshold=3, reference_pixel="center"):
        """Returns an aperture mask creating using the thresholding method.

        Compute for only the single cadence selected.

        This method will identify the pixels in the TargetPixelFile which show
        a median flux that is brighter than `threshold` times the standard
        deviation above the flux values. The standard deviation is estimated
        in a robust way by multiplying the Median Absolute Deviation (MAD)
        with 1.4826.

        If the thresholding method yields multiple contiguous regions, then
        only the region closest to the (col, row) coordinate specified by
        `reference_pixel` is returned.  For exmaple, `reference_pixel=(0, 0)`
        will pick the region closest to the bottom left corner.
        By default, the region closest to the center of the mask will be
        returned. If `reference_pixel=None` then all regions will be returned.

        Parameters
        ----------
        cadenceno : int
            Cadence index to compute threshold mask for
        threshold : float
            A value for the number of sigma by which a pixel needs to be
            brighter than the median flux to be included in the aperture mask.
        reference_pixel: (int, int) tuple, 'center', or None
            (col, row) pixel coordinate closest to the desired region.
            For example, use `reference_pixel=(0,0)` to select the region
            closest to the bottom left corner of the target pixel file.
            If 'center' (default) then the region closest to the center pixel
            will be selected. If `None` then all regions will be selected.

        Returns
        -------
        aperture_mask : ndarray
            2D boolean numpy array containing `True` for pixels above the
            threshold.
        """
        if reference_pixel == "center":
            reference_pixel = (self.shape[2] / 2, self.shape[1] / 2)
        # Find the requested cadence index
        cadence_idx = np.nonzero(np.in1d(self.cadenceno, cadenceno))[0]
        image = np.array(self.flux[cadence_idx])[0]
        vals = image[np.isfinite(image)].flatten()
        # Calculate the theshold value in flux units
        mad_cut = (1.4826 * median_absolute_deviation(vals) * threshold) + np.nanmedian(image)
        # Create a mask containing the pixels above the threshold flux
        threshold_mask = np.nan_to_num(image) >= mad_cut
        if (reference_pixel is None) or (not threshold_mask.any()):
            # return all regions above threshold
            return threshold_mask
        else:
            # Return only the contiguous region closest to `region`.
            # First, label all the regions:
            labels = label(threshold_mask)[0]
            # For all pixels above threshold, compute distance to reference pixel:
            label_args = np.argwhere(labels > 0)
            distances = [
                np.hypot(crd[0], crd[1])
                for crd in label_args
                - np.array([reference_pixel[1], reference_pixel[0]])
            ]
            # Which label corresponds to the closest pixel?
            closest_arg = label_args[np.argmin(distances)]
            closest_label = labels[closest_arg[0], closest_arg[1]]
            return labels == closest_label

