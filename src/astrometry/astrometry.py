""" Tools for performing astrometry analysis on asteroids in TESS data"""

from lightkurve.targetpixelfile import TessTargetPixelFile
import lightkurve as lk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.stats import median_absolute_deviation
from scipy.interpolate import PchipInterpolator
from astropy.io import fits
from tqdm import tqdm
import csv
from scipy.optimize import minimize_scalar


#*************************************************************************************************************
class Centroids:
    """ Object to store computed centroid information

    Atributes
    ---------
    timestamps  : [np.array]
        Timestamps in BTJD
    col         : [np.array]
        Column centroids in pixels
    row         : [np.array]
        Row centroids in pixels
    ra          : [np.array]
        Right Ascension centroids in degrees
    dec         : [np.array]
        Declination centroids in degrees
    colRowData : bool
        If column and row centroid data in pixels provided
    raDecData : bool
        If RA and Decl. centroid data in pixels provided
    """

    def __init__(self, timestamps, col=None, row=None, ra=None, dec=None):
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

        self.row = np.array(row)
        self.col = np.array(col)

        self.ra     = np.array(ra)
        self.dec    = np.array(dec)

    def write_to_file(self, filename):
        """ Writes out the centroids to a CSV file
        """

        if self.colRowData:
            rows = [[self.row[idx], self.col[idx], self.timestamps[idx]] for idx in np.arange(len(centroids.row))]
        elif self.raDecData:
            rows = [[self.ra[idx], self.dec[idx], self.timestamps[idx]] for idx in np.arange(len(centroids.ra))]


        with open(filename, 'w') as fp:
            wr = csv.writer(fp)
            wr.writerows(rows)

    def __repr__(self):
        return self.__dict__
    

#*************************************************************************************************************
def compute_centroids(mtpf, method='quadratic', CCD_ref=False, image_reference=False, aper_mask=10.0):
    """ Computes the centroid of a moving target pixel file

    Parameters
    ----------
    mtpf : lightkurve.targetpixelfile.MovingTargetPixelFile
            The moving target pixel file to use
    method : str
            The centroiding method to use: 'moments' 'quadratic'
    CCD_ref : bool
            If True then add in the mtpf CORNER_COLUMN and CORNER_ROW to get the CCD reference pixel coordinates
    image_reference : bool
            If True then subtract off 0.5 so that the pixel center is at 0.0 
            This aids when superimposing on the pixel grid with matplot lib
            (The TESS convenction is the center is at 0.5)
    aper_mask : float
        The aperture masking threshold

    Returns
    -------
    centroids : astrometry.Centroids class
        The computed centroids

    """

    # Aperture: Compute the mean flux per pixel and then threshold at a flux value of 10
    aper = mtpf.flux.mean(axis=0).value > aper_mask
    cols, rows = mtpf.estimate_centroids(method=method, aperture_mask=aper)

    # Subtract off the extra 0.5 so that the centroid is plotted properly on a pixel grid in matplotlib
    if image_reference:
        cols -= 0.5*u.pixel
        rows -= 0.5*u.pixel
    
    if CCD_ref:
        cols += mtpf.hdu[1].data['CORNER_COLUMN'][mtpf.quality_mask]*u.pixel
        rows += mtpf.hdu[1].data['CORNER_ROW'][mtpf.quality_mask]*u.pixel
        pass


    centroids = Centroids(mtpf.time, col=cols, row=rows)

    return centroids

#*************************************************************************************************************
def detrend_centroids_via_poly (mtpf, centroids, polyorderRange=[1,8], sigmaThreshold=5.0, remove_expected=False, fig=None, debug=False):
    """ Detrends any trends in the centroid motion via peicewise polynomial fitting.

    This function will potioinally first remove the JPL Horizona expected centroids if requested.

    Then it will identify discontinuities (i.e. orbit boundaries) or any other spurious regions. It will
    then chunk the data around discontinuities and gap the spurious regions. It will then fit the curves to a dynamic order
    polynomial, where the optimal polyorder is chosen based on RMSE.

    This function will work on either row/col centroids or ra/dec centroids.

    Parameters
    ----------
    mtpf : lightkurve.targetpixelfile.MovingTargetPixelFile
            The moving target pixel file to use
    centroids : astrometry.Centroids class
        The computed centroids
    polyorderRange : ndarray list(2)
        The upper and lower polyorders to try
    sigmaThreshold : float
        The sigma threshold for finding segments

    Returns
    -------
    detrended_centroids : astrometry.Centroids class
        The detrended centroids


    """

    #***
    # First remove the JPL Horizons expected centroids, if requested
    if remove_expected:
        col_exp =mtpf.hdu[1].data['TARGET_COLUMN'][mtpf.quality_mask]
        row_exp =mtpf.hdu[1].data['TARGET_ROW'][mtpf.quality_mask]
      # centroids.col -= col_exp
      # centroids.row -= row_exp

      # test = remove_expected_trend_elemental(centroids.row, row_exp)
      # test = remove_expected_trend_elemental(centroids.col, col_exp)

    #***
    # Peicewise polynomial fit
    if centroids.colRowData:
        detrended_col_centroids, col_polyFitCurve, col_break_point = _detrend_centroids_elemental(centroids.col, mtpf.time,
                mtpf.cadenceno, polyorderRange, sigmaThreshold=sigmaThreshold)
        detrended_row_centroids, row_polyFitCurve, row_break_point = _detrend_centroids_elemental(centroids.row, mtpf.time,
                mtpf.cadenceno, polyorderRange, sigmaThreshold=sigmaThreshold)
        
        detrended_centroids = Centroids(mtpf.time, col=detrended_col_centroids, row=detrended_row_centroids)
    elif centroids.raDecData:
        detrended_ra_centroids, ra_polyFitCurve, ra_break_point = _detrend_centroids_elemental(centroids.ra, mtpf.time,
                mtpf.cadenceno, polyorderRange, sigmaThreshold=sigmaThreshold)
        detrended_dec_centroids, dec_polyFitCurve, dec_break_point = _detrend_centroids_elemental(centroids.dec, mtpf.time,
                mtpf.cadenceno, polyorderRange, sigmaThreshold=sigmaThreshold)
        
        detrended_centroids = Centroids(mtpf.time, ra=detrended_ra_centroids, dec=detrended_dec_centroids)


    # Plot the detrended centroids
    # We can plot either row/col or ra/dec
    if debug:
        if fig is None:
            fig,ax = plt.subplots(1,1, figsize=(12, 10))
        else:
            fig.clf()

        #***
        # Raw centroids
        ax = plt.subplot(4,1,1)
        if centroids.colRowData:
            ax.plot(mtpf.time.value, centroids.col, '*b', label='Column Centroids')
            ax.plot(mtpf.time.value[col_break_point], centroids.col[col_break_point], '*r', markersize=10, label='Column Breakpoints')
            ax.plot(mtpf.time.value, col_polyFitCurve, '-m', label='Column PolyFit')
            ax.plot(mtpf.time.value, centroids.row, '*c', label='Row Centroids')
            ax.plot(mtpf.time.value[row_break_point], centroids.row[row_break_point], '*r', markersize=10, label='Row Breakpoints')
            ax.plot(mtpf.time.value, row_polyFitCurve, '-m', label='Row PolyFit')
            minVal = np.nanmin([centroids.col, centroids.row])
            maxVal = np.nanmax([centroids.col, centroids.row])
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
        dumpHere = np.nonzero(mtpf.hdu[1].data['QUALITY'] & 32 > 0)[0]
       #for idx in dumpHere:
       #    ax.plot([mtpf.hdu[1].data['TIME'][dumpHere], mtpf.hdu[1].data['TIME'][dumpHere]], [minVal, maxVal], '-k')
        ax.vlines(mtpf.time.value[dumpHere], ymin=minVal, ymax=maxVal, colors='k', label='Momentum Dumps')
        ax.legend()
        ax.set_title('Removing long term trends in centroids')
        ax.grid()

        #***
        # Residual from polynomial
        ax = plt.subplot(4,1,2)
        if centroids.colRowData:
            col = detrended_centroids.col
            row = detrended_centroids.row
            ax.plot(mtpf.time.value, col, '*-b', label='Column Residual; std={:.3f}'.format(np.nanstd(col)))
            ax.plot(mtpf.time.value, row, '*-c', label='Row Residual; std={:.3f}'.format(np.nanstd(row)))
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
        ax.vlines(mtpf.time.value[dumpHere], ymin=minVal, ymax=maxVal, colors='k', label='Momentum Dumps')
        # Plot where the pixel grid snaps a pixel
        ax.legend()
        ax.set_title('Centroid Residuals')
        ax.grid()

        #***
        # Residula from demoding
        ax = plt.subplot(4,1,3)
        if centroids.colRowData:
            col = demoded_centroids.col
            row = demoded_centroids.row
            ax.plot(mtpf.time.value, col, '*-b', label='Column Demoded; std={:.3f}'.format(np.nanstd(col)))
            ax.plot(mtpf.time.value, row, '*-c', label='Row Demoded; std={:.3f}'.format(np.nanstd(row)))
            ax.set_ylabel('Pixels')
            minVal = np.nanmin([col, row])
            maxVal = np.nanmax([col, row])
        elif centroids.raDecData:
            # plot ra and dec in arcseconds
            ra  = demoded_centroids.ra * 60**2
            dec = demoded_centroids.dec * 60**2
            ax.plot(mtpf.time.value, ra, '*-b', label='R.A. Demoded; std={:.3f}'.format(np.nanstd(ra)))
            ax.plot(mtpf.time.value, dec, '*-c', label='Decl. Demoded; std={:.3f}'.format(np.nanstd(dec)))
            ax.set_ylabel('Arcseconds')
            minVal = np.nanmin([ra, dec])
            maxVal = np.nanmax([ra, dec])
       #for idx in dumpHere:
       #    ax.plot([mtpf.hdu[1].data['TIME'][dumpHere], mtpf.hdu[1].data['TIME'][dumpHere]], [minVal, maxVal], '-k')
       #ax.vlines(mtpf.time.value[dumpHere], ymin=minVal, ymax=maxVal, colors='k', label='Momentum Dumps')
        # Plot where the pixel grid snaps a pixel
        ax.legend()
        ax.set_title('Centroid Demoded')
        ax.grid()

        #***
        # Examine the Periodogram
        # Convert centroids to LightCurve objects
        ax = plt.subplot(4,1,4)
        if centroids.colRowData:
            col_lc = lk.LightCurve(time=mtpf.time, flux=detrended_centroids.row)
            row_lc = lk.LightCurve(time=mtpf.time, flux=detrended_centroids.col)
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
        


    return detrended_centroids 
    

#*************************************************************************************************************
def detrend_centroids_expected_trend(mtpf, centroids):
    """ Detrends the centroids using the given expected trends
    """

    # The JPL Horixons expected locations
    col_exp = mtpf.hdu[1].data['TARGET_COLUMN'][mtpf.quality_mask]
    row_exp = mtpf.hdu[1].data['TARGET_ROW'][mtpf.quality_mask]

    rowRemoved, rowExpRemoved, rowFittedTrend = _remove_expected_trend_elemental(centroids.row, row_exp)
    colRemoved, colExpRemoved, colFittedTrend = _remove_expected_trend_elemental(centroids.col, col_exp)

    detrended_centroids = Centroids(mtpf.time, col=colRemoved, row=rowRemoved)

    # Now print the results
    fig,ax = plt.subplots(1,1, figsize=(12, 10))

    # Initial and expected centroids
    ax = plt.subplot(4,1,1)
    ax.plot(mtpf.time.value, centroids.col, '*b', label='Column Centroids')
    ax.plot(mtpf.time.value, col_exp, '-m', label='Column Expected')
    ax.plot(mtpf.time.value, centroids.row, '*c', label='Row Centroids')
    ax.plot(mtpf.time.value, row_exp, '-m', label='Row Expected')
    plt.legend()
    plt.title('Raw Centroids and JPL Horozons Expected')
    plt.grid()

    # Removing the moded motion
    ax = plt.subplot(4,1,2)
    ax.plot(mtpf.time.value, colExpRemoved, '*b', label='Column Centroids')
    ax.plot(mtpf.time.value, colFittedTrend + np.nanmean(colExpRemoved), '-m', label='Column Fitted Modded Trend')
    ax.plot(mtpf.time.value, rowExpRemoved, '*c', label='Row Centroids')
    ax.plot(mtpf.time.value, rowFittedTrend + np.nanmean(rowExpRemoved), '-m', label='Row Fitted Moded Trend')
    plt.legend()
    plt.title('With Expected Removed')
    plt.grid()

    # Final Residual
    ax = plt.subplot(4,1,3)
    ax.plot(mtpf.time.value, colRemoved, '*b', label='Column Residual; std={:.3f}'.format(np.nanstd(colRemoved)))
    ax.plot(mtpf.time.value, rowRemoved, '*c', label='Row Residual; std={:.3f}'.format(np.nanstd(colRemoved)))
    plt.legend()
    plt.title('Final Residual')
    plt.grid()

    # Periodigram


#*************************************************************************************************************
def _remove_expected_trend_elemental(centroid, exp_trend, subtract_mean=True):
    """ Removes the expected motion trend from the centroids

    Parameters
    ----------
    subtract_mean : bool
        If True then subtract the secular to so the mean is zero

    Returns
    -------
    detrended_centroid : np.array
        The centroid with both the expected curve and the moded sawtooth removed
    centroid_exp_removed : np.array
        The centroid with just the expected curve removed
    fittedTrend : np.array
        The Fit to the moded trend

    """


    # First remove the first-order term
    centroid_exp_removed = centroid - exp_trend

    # Remove the secular term in the residual
    centroid_exp_removed -= np.nanmean(centroid_exp_removed) 

    # Now remove the moded motion
    moded_trend = np.mod(exp_trend, 1)
    moded_trend = (moded_trend / np.mean(moded_trend)) - 1
    # Perform a simple least-squares fit


    #***
    # Use scipy.optimize.minimize_scalar
    # Minimize the RMSE
    def rmse(coeff):
        return np.sqrt(np.sum((centroid_exp_removed - moded_trend*coeff)**2))

    minimize_result = minimize_scalar(rmse, method='Bounded',
            bounds=[-1,1])
  #         options={'maxiter':max_iter, 'disp': False})

    fittedTrend = moded_trend*minimize_result.x
    detrended_centroid = centroid_exp_removed - fittedTrend 

    return detrended_centroid, centroid_exp_removed, fittedTrend

#*************************************************************************************************************
def _detrend_centroids_elemental(centroids, time, cadenceno, polyorderRange, sigmaThreshold=5.0):
    """ Detrends a single centroids time series.

    Helper function for detrend_centroidss 

    Parameters
    ----------
    centroids : np.array list
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
    detrended_centroids  : np.array list 
    polyFitCurve    : np.array list
        The fitted polynomial curve
    break_point  : int list
        The cadence indices where breaks occur
    
    """

    # Use first differences to find abrupt changes in the centroids
    # First fill gaps using PCHIP interpolation
    # We need to fill timestamps too
    filled_cadenceno = np.arange(cadenceno[0], cadenceno[-1]+1)
    # Fill the centroids to the filled timstamps
    fInterpPCHIP = PchipInterpolator(cadenceno, centroids, extrapolate=False)
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
    centroidssDetrended = np.full(np.shape(centroids), np.nan)
    polyFitCurve        = np.full(np.shape(centroids), np.nan)
    centroidsDetrended  = np.full(np.shape(centroids), np.nan)
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
            polyFitCurve[chunkStart] = centroids[chunkStart]
        else:
            polyFitCurve[chunkStart:chunkEnd], _ = _find_best_polyfit(time[chunkStart:chunkEnd].value, centroids[chunkStart:chunkEnd], polyorderRange)


    detrended_centroids = centroids - polyFitCurve

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

    
   #def __init__(self, path, quality_bitmask="default", **kwargs):
   #    """ See the TargetPixelFile __init__ for arguments
   #    """

   #    # Call the TargetPixelFile Constructor
   #    super(TargetPixelFileAstrometry, self).__init__(path, quality_bitmask=quality_bitmask, **kwargs)

    def remove_barycenter_correction(self):
        """ The target pixel files are barycenter corrected. This method will remove that correctin using 
        the TIMECORR column in the FITs file.
        """

        return


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
        self, step: int = None, interval: int = 200, extra_data=None, save_file=None, fps=30, **plot_args
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
        extra_data : float np.array(2,:)
            Extra scatter data to plot with the pixel data
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
        if (extra_data is not None):
            sct = ax.scatter(extra_data[0,step], extra_data[1,step], c='k')

        def init():
            return ax.images

        def animate(i):
            frame = i * step
            ax.images[0].set_data(self.hdu[1].data[column][self.quality_mask][frame])
            if (extra_data is not None):
                sct.set_offsets(np.array([extra_data[0,frame], extra_data[1,frame]]))
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

