""" Tools for performing astrometry analysis on asteroids in TESS data"""

import warnings
from lightkurve.targetpixelfile import TessTargetPixelFile
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.interpolate import PchipInterpolator
from astropy.io import fits
from tqdm import tqdm
import csv
from scipy.optimize import minimize_scalar
from tess_ephem import ephem
from scipy.stats import median_abs_deviation
import os
import glob
import pandas as pd
from matplotlib import patches, animation
#import multiprocessing as mp

from MovingTargetPixelFile import MovingTargetPixelFile


#*************************************************************************************************************
class MovingCentroids:
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
        """ Generate the MovingCentroids object. 
        
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
        """ The instrument time given by the mtpf"""

        return self.mtpf.instrument_time

    #*************************************************************************************************************
    def download_expected_motion(self, aberrate : bool = False, use_mtpf_stored : bool = False):
        """ Downloads the expected motion in R.A. and decl. using tess-ephem, which uses JPL Horizons

        NOTE: By default tess-ephem will perform the approximate DVA correction when returning the column and row
        coordinates. The <aberrate> argument will enable or disable this DVA correction.
        The RA and Decl. returned by ephem are the true coords, irrespective of the <aberrate> value.

        Parameters
        ----------
        aberrate : bool
            If true then apply the approximate DVA correction for the returned row and column
        use_mtpf_stored : bool
            The ecpected astrometry is stored in the mtpf, just return that instead.
            Note that this only includes the row and column data

        """

        if use_mtpf_stored:
            self.expected_row = self.mtpf.hdu[1].data['TARGET_ROW'][self.mtpf.quality_mask]
            self.expected_col = self.mtpf.hdu[1].data['TARGET_COLUMN'][self.mtpf.quality_mask]

            # These are not available in the mtpf
            self.expected_ra = np.full(len(self.time), np.nan)
            self.expected_dec = np.full(len(self.time), np.nan)

        else:
            # Download the JPL Horizons data at the data cadence times
          # print('Downloading Expected R.A. and decl. from JPL Horizons...')
            df = ephem(self.mtpf.targetid, time=self.time, interpolation_step='2m', verbose=True, aberrate=aberrate)
         
            # Pad invalid times with NaN
            dfTimeArray = [t.value for t in df.index]
            presentTimes = np.nonzero(np.in1d(self.time.value, dfTimeArray))[0]
            self.expected_ra = np.full(len(self.time), np.nan)
            self.expected_ra[presentTimes] = df.ra.values
            self.expected_dec = np.full(len(self.time), np.nan)
            self.expected_dec[presentTimes] = df.dec.values
         
            self.expected_row = np.full(len(self.time), np.nan)
            self.expected_row[presentTimes] = df.row.values
            self.expected_col = np.full(len(self.time), np.nan)
            self.expected_col[presentTimes] = df.column.values

    #*************************************************************************************************************
    def compute_centroids_simple_aperture(self, method='moments', CCD_ref=True, aper_mask_threshold=3.0):
        """ Computes the centroid of a moving target pixel file using a simple static aperture
        
        Parameters
        ----------
        method : str
                The centroiding method to use: 'moments' 'quadratic'
        CCD_ref : bool
                If True then add in the mtpf CORNER_COLUMN and CORNER_ROW to get the CCD reference pixel coordinates
        aper_mask_threshold : float
            A value for the number of sigma by which a pixel needs to be
            brighter than the median flux to be included in the aperture mask.
        
        Returns
        -------
        centroidsMatrix : float np.array(nCadences,2)
            Centroid data relative to mask for use with self.mtpf.animate method

        And also modifies these class attributes:
        self.row : array of row centroids
        self.col : array of column centroids
        self.colRowDataAvailable = True

        
        """
        
        # Calculate the median image
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            median_image = np.array(np.nanmedian(self.mtpf.flux, axis=0))
        vals = median_image[np.isfinite(median_image)].flatten()
        # Calculate the theshold value in flux units
        mad_cut = (1.4826 * median_abs_deviation(vals) * aper_mask_threshold) + np.nanmedian(median_image)
        # Create a mask containing the pixels above the threshold flux
        aper = np.nan_to_num(median_image) >= mad_cut
        cols, rows = self.mtpf.estimate_centroids(method=method, aperture_mask=aper)
        
        centroidsMatrix = np.transpose([cols, rows])

        if CCD_ref:
            cols += self.mtpf.corner_column * u.pixel
            rows += self.mtpf.corner_row * u.pixel
        
        self.col = cols
        self.row = rows

        self.colRowDataAvailable = True
        
        return centroidsMatrix

    #*************************************************************************************************************
    def compute_centroids_dynamic_aperture(self, method='moments', CCD_ref=True,
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
        aper_mask_threshold : float
            A value for the number of sigma by which a pixel needs to be
            brighter than the median flux to be included in the aperture mask.
        n_cores : int
            Number of multiprocessing cores to use. None means use all.

        Returns
        -------
        centroidsMatrix : float np.array(nCadences,2)
            Centroid data relative to mask for use with self.mtpf.animate method

        And also modifies these class attributes:
        self.row : array of row centroids
        self.col : array of column centroids
        self.colRowDataAvailable = True

        """

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
        

        # Compute the aperture for each cadence seperately
        # TODO: This is ripe for parallelization. Figure out how to do that within a class object
        for idx,cadenceno in enumerate(self.mtpf.cadenceno):
            self.aper[idx,:,:] = self.mtpf.create_threshold_mask_one_cadence(cadenceno, threshold=aper_mask_threshold)
            col, row = self.mtpf.estimate_centroids_one_cadence(cadenceno, method=method,
                    aperture_mask=self.aper[idx,:,:])
            cols.append(col)
            rows.append(row)

        cols = np.array(cols).flatten() * u.pixel
        rows = np.array(rows).flatten() * u.pixel

        centroidsMatrix = np.transpose([cols, rows])

        if CCD_ref:
            cols += self.mtpf.corner_column * u.pixel
            rows += self.mtpf.corner_row * u.pixel
        
        self.col = cols
        self.row = rows

        self.colRowDataAvailable = True
        
        return centroidsMatrix
        
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
                'raDec2Pix_ra_dec': The ra and dec computed by the SPOC raDec2Pix class
        filename : str
            Name of file to read from

        Returns
        -------
        if data='raDec2Pix_ra_dec':
            self.raDec2Pix_ra 
            self.raDec2Pix_dec 
        """

        assert data=='raDec2Pix_ra_dec', 'data=raDec2Pix_ra_dec is the only current option'

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

        self.raDecDataAvailable = True


    #*************************************************************************************************************
    def detrend_centroids_via_poly (self, polyorderRange=[1,8], sigmaThreshold=5.0, remove_expected=False,
            include_DVA=False, fig=None, plot=False):
        """ Detrends any trends in the centroid motion via piecewise polynomial fitting.
 
        This function will optionally first remove the JPL Horizons expected centroids if requested.
 
        Then it will identify discontinuities (i.e. orbit boundaries) or any other spurious regions. It will
        then chunk the data around discontinuities and gap the spurious regions. It will then fit the curves to a dynamic order
        polynomial, where the optimal polyorder is chosen based on RMSE.
 
        This function will work on either row/col centroids or ra/dec centroids depending on what data is available.
 
        Parameters
        ----------
        polyorderRange : ndarray list(2)
            The upper and lower polyorders to try
        sigmaThreshold : float
            The sigma threshold for finding segments
        remove_expected : bool
            If True then first subtract off the JPL Horizons expected astrometry
        include_DVA : bool
            If True then use includes the DVA term when computing the JPL Horizons expected trend
        fig : figure handle
            If passed (and plot==True) then plot on this figure
        plot : bool
            If True then generate plot
 
        Returns
        -------
        column, row : `~astropy.units.Quantity`, `~astropy.units.Quantity`
            Floats containing the column and row detrended centroids
 
 
        """
 
        #***
        # First remove the JPL Horizons expected centroids, if requested
        if remove_expected:
            self.download_expected_motion(aberrate=include_DVA)
            rowExpRemoved = _remove_expected_trend_elemental(self.row, self.expected_row)
            colExpRemoved = _remove_expected_trend_elemental(self.col, self.expected_col)
            self.col -= colExpRemoved 
            self.row -= rowExpRemoved 
 
        #***
        # Piecewise polynomial fit
        if self.colRowDataAvailable:
            detrended_col_centroids, col_polyFitCurve, col_break_point = _detrend_centroids_elemental(self.col, self.mtpf.time,
                    self.mtpf.cadenceno, polyorderRange, sigmaThreshold=sigmaThreshold)
            detrended_row_centroids, row_polyFitCurve, row_break_point = _detrend_centroids_elemental(self.row, self.mtpf.time,
                    self.mtpf.cadenceno, polyorderRange, sigmaThreshold=sigmaThreshold)
            
        elif self.raDecDataAvailable:
            raise Exception('This option has not been maintained')
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
            elif centroids.raDecDataAvailable:
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
                madstd = lambda x: 1.4826*median_abs_deviation(x, nan_policy='omit')
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
    def detrend_centroids_expected_trend(self, include_DVA=False, extra_title="", plot=False):
        """ Detrends the centroids using the given the JPL Horizons expected trends

        Parameters
        ----------
        include_DVA : bool
            If True then use includes the DVA term when computing the JPL Horizons expected trend
        extra_title : str
            Extra title to prepend to figure title
        plot : bool
            If True then generate plot
        """
 
        # The JPL Horizons expected astrometry
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
        
            # Final Residual
            ax = plt.subplot(2,1,2)
            madstd = lambda x: 1.4826*median_abs_deviation(x, nan_policy='omit')
            ax.plot(self.mtpf.time.value, colExpRemoved, '*b', label='Column Residual; madstd={:.3f}'.format(madstd(colExpRemoved)))
            ax.plot(self.mtpf.time.value, rowExpRemoved, '*c', label='Row Residual; madstd={:.3f}'.format(madstd(rowExpRemoved)))
            # Set plot limits to ignore excursions
            allYData = np.concatenate((colExpRemoved, rowExpRemoved))
          # yUpper = np.nanpercentile(allYData, 99.5)
          # yLower = np.nanpercentile(allYData, 0.5)
            yUpper = np.nanpercentile(allYData, 99.0)
            yLower = np.nanpercentile(allYData, 1.0)
            ax.set_ylim(yLower, yUpper)
          # ax.set_ylim(-1.0, 1.0)
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
    def compute_centroid_proper_motion(centroids):
        """ This will compute the proper motion of the tartet centroids.
 
        It simply computes the average motion as distance per time in units of pixels per day
 
        Parameters
        ----------
        centroids : MovingCentroids class
            col         : [np.array]
                Column centroids in pixels
            row         : [np.array]
                Row centroids in pixels
            time  : astropy.time.core.Time
                Timestamps in BTJD
 
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
# Define the elemental process for each process in the compute_centroids_dynamic_aperture pool
def _single_cadence_dynamic_aperture (mtpf, aper_mask_threshold, method, idx, cadenceno):
    aper = mtpf.create_threshold_mask_one_cadence(cadenceno, threshold=aper_mask_threshold)
    col, row = mtpf.estimate_centroids_one_cadence(cadenceno, method=method,
            aperture_mask=aper)
    return aper, col, row

#*************************************************************************************************************
def _remove_expected_trend_elemental(centroid, exp_trend, subtract_mean=True):
    """ Removes the expected motion trend from the centroids

    This is experimental but right now all it does is subtract off the expected trend from the centroids. Other code is
    commented out.

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
    colSigma = 1.4826*median_abs_deviation(diffsNorm)
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
    """ Finds the best polyorder to minimize RMS with the <polyOrderRange>

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
    centroids : list of astrometry.MovingCentroids class
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

