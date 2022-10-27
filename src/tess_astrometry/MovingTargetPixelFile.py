""" MovingTargetPixelFile class moving targets in TESS FOV
"""

import warnings
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from scipy.ndimage import label
from matplotlib import patches, animation

from lightkurve.utils import validate_method
from lightkurve.targetpixelfile import TessTargetPixelFile
from lightkurve.lightcurve import LightCurve, TessLightCurve

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

    def __repr__(self):
        return "MovingTargetPixelFile(Target ID: {})".format(self.targetid)

    @property
    def camera(self):
        """ The camera is an array """
        return self.hdu[1].data['CAMERA'][self.quality_mask]

    @property
    def ccd(self):
        """ The ccd is an array """
        return self.hdu[1].data['CCD'][self.quality_mask]

    @property
    def column(self):
        """The relative column index for the mask (I.e. 0)"""
        return 0

    @property
    def row(self):
        """The relative row index for the mask (I.e. 0)"""
        return 0

    @property
    def corner_column(self):
        """CCD reference pixel column number ('CORNER_COLUMN' header keyword)."""
        return self.hdu[1].data['CORNER_COLUMN'][self.quality_mask]

    @property
    def corner_row(self):
        """CCD pixel row number ('CORNER_ROW' header keyword)."""
        return self.hdu[1].data['CORNER_ROW'][self.quality_mask]

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
        self, step: int = None, interval: int = 200, centroidsMatrix=None, aperture_mask=None, save_file=None, fps=30, **plot_args
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
        centroidsMatrix : float np.array(nCadences,2)
            Centroid data to plot overlaid on the pixel data
            Be sure to use centroids relative to corner of mask
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
        if centroidsMatrix is not None:
            sct = ax.scatter(centroidsMatrix[step, 0], centroidsMatrix[step, 1], c='k')

        def add_aper(aperture_mask_single, ax):
            mask_color="red"
            aperture_mask_single = self._parse_aperture_mask(aperture_mask_single)
            # Remove any current patches to reset the aperture mask in the animation
          # if ax.patches != []:
          #     ax.patches = []
            if len(ax.patches) > 0:
                for patch in ax.patches:
                    ax.patches.pop()
                ax.figure.canvas.draw()
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
            if centroidsMatrix is not None:
                sct.set_offsets(np.array([centroidsMatrix[frame, 0], centroidsMatrix[frame, 1]]))
            if aperture_mask is not None:
                add_aper(aperture_mask[frame,:,:], ax)
            ax.set_title(f"Frame {frame}")
            return

        plt.close(ax.figure)  # prevent figure from showing up in interactive mode

        # `blit=True` means only re-draw the parts that have changed.
        frames = len(self) // step
        anim = animation.FuncAnimation(
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
        yy, xx = np.indices(self.shape[1:])
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
        col_centr = np.asfarray(col_centr)
        row_centr = np.asfarray(row_centr)
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
        mad_cut = (1.4826 * median_abs_deviation(vals) * threshold) + np.nanmedian(image)
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

    def extract_dynamic_aperture_photometry(
        self, aperture_mask=None, flux_method="sum", centroid_matrix=None
    ):
        """Returns a LightCurve obtained using aperture photometry using a different aperture for each cadence.

        Parameters
        ----------
        aperture_mask : array-like
            A boolean array of shape (n_cadences, col_pixels, row_pixels) describing the aperture such that `True` means
            that the pixel will be used. n_cadences must be the same length as cadences in the mtpf
        flux_method: 'sum', 'median', or 'mean'
            Determines how the pixel values within the aperture mask are combined
            at each cadence. Defaults to 'sum'.
        centroid_matrix : float np.array(nCadences,2)
            from MovingCentroids.compute_centroids_dynamic_aperture

        Returns
        -------
        lc : TessLightCurve object
            Contains the summed flux within the aperture for each cadence.
        """

        assert aperture_mask is not None, 'aperture_mask must be passed'

        flux, flux_err = self._dynamic_aperture_photometry(
            aperture_mask=aperture_mask,
            flux_method=flux_method,
        )
        keys = {
                "centroid_col": centroid_matrix[:,0],
            "centroid_row": centroid_matrix[:,1],
            "quality": self.quality,
            "sector": self.sector,
            "camera": self.camera,
            "ccd": self.ccd,
            "mission": self.mission,
            "cadenceno": self.cadenceno,
            "ra": self.ra,
            "dec": self.dec,
            "label": self.get_keyword("OBJECT", default=self.targetid),
            "targetid": self.targetid,
        }
        meta = {"APERTURE_MASK": aperture_mask}
        return TessLightCurve(
            time=self.time, flux=flux, flux_err=flux_err, **keys, meta=meta
        )

    def _dynamic_aperture_photometry(
        self, aperture_mask, flux_method="sum"
    ):
        """Helper method for ``extract_aperture photometry``.

        Returns
        -------
        flux, flux_err
        """

        # Aperture mask shoudl be booleans, not integers
        aperture_mask = aperture_mask.astype(bool)

        # Estimate flux
        flux = u.Quantity(np.full(self.flux[:,0,0].shape, np.nan), unit="electron/s")
        if flux_method == "sum":
            for idx in np.arange(len(self.cadenceno)):
                flux[idx] = np.nansum(self.flux[idx, aperture_mask[idx,:,:]])

        elif flux_method == "median":
            for idx in np.arange(len(self.cadenceno)):
                flux[idx] = np.nanmedian(self.flux[idx, aperture_mask[idx,:,:]])

        elif flux_method == "mean":
            for idx in np.arange(len(self.cadenceno)):
                flux[idx] = np.nanmean(self.flux[idx, aperture_mask[idx,:,:]])
        else:
            raise ValueError("`flux_method` must be one of 'sum', 'median', or 'mean'.")

        # We use ``np.nansum`` above to be robust against a subset of pixels
        # being NaN, however if *all* pixels are NaN, we propagate a NaN.
        is_allnan = np.full(len(flux), False)
        for idx in np.arange(len(flux)):
            is_allnan[idx] = ~np.any(np.isfinite(self.flux[idx, aperture_mask[idx,:,:]]))
        flux[is_allnan] = np.nan

        # Similarly, if *all* pixel values across the TPF are exactly zero,
        # we propagate NaN (cf. #873 for an example of this happening)
        is_allzero = np.all(self.flux == 0, axis=(1, 2))
        flux[is_allzero] = np.nan

        # Estimate flux_err
        with warnings.catch_warnings():
            # Ignore warnings due to negative errors
            warnings.simplefilter("ignore", RuntimeWarning)
            flux_err = u.Quantity(np.full(self.flux_err[:,0,0].shape, np.nan), unit="electron/s")
            if flux_method == "sum":
                for idx in np.arange(len(self.cadenceno)):
                    flux_err[idx] = np.nansum(self.flux_err[idx, aperture_mask[idx,:,:]] ** 2.0) ** 0.5

            elif flux_method == "median":
                #TODO: Do we really take a median in quadrature here?!?!
                # This si what the original _aperture_photometry code in targetpixelfile.py does but doesn't seem right
                # to me. Probably doesn't matter since we'll never use median or mean.
                for idx in np.arange(len(self.cadenceno)):
                    flux_err[idx] = np.nanmedian(self.flux_err[idx, aperture_mask[idx,:,:]] ** 2.0) ** 0.5

            elif flux_method == "mean":
                for idx in np.arange(len(self.cadenceno)):
                    flux_err[idx] = np.nanmean(self.flux_err[idx, aperture_mask[idx,:,:]] ** 2.0) ** 0.5

            is_allnan = np.full(len(flux), False)
            for idx in np.arange(len(flux)):
                is_allnan[idx] = ~np.any(np.isfinite(self.flux_err[idx, aperture_mask[idx,:,:]]))
            flux_err[is_allnan] = np.nan

        if self.get_header(1).get("TUNIT5") == "e-/s":
            flux = u.Quantity(flux, unit="electron/s")
        if self.get_header(1).get("TUNIT6") == "e-/s":
            flux_err = u.Quantity(flux_err, unit="electron/s")

        return flux, flux_err


