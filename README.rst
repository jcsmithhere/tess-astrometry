tess-astrometry
===============

**Tools to measure the astrometry of TESS solar-system objects.**


This package contains two modules: 

* `MovingTargetPixelFiles` -- Generates a moving TargetPixelFile (mtpf) object for use with the output of
  `tess_cloud.asteroid_pipeline <https://github.com/SSDataLab/tess-cloud>`_ FITs files. 
* `astrometry` -- Tools to measure the centroids of the moving target pisel files and compare them to the expected
  astrometry given by JPL Horizons.


Usage
-----

To read in a mtpf FITS file:

.. code-block:: python

    >>>from astropy.io import fits
    >>>from tess_astrometry.MovingTargetPixelFile import MovingTargetPixelFile
    >>>f = fits.open(file_path)
    >>>mtpf = ast.MovingTargetPixelFile(f, quality_bitmask="hardest")

the `mtfp` inherents `lightkurve.TargetPixelFile` but adds functionality specific to moving objects.

Measuring centroids is via use of the `MovingCentroids` class. There are two methods to compute the centroids:

#. `compute_centroids_simple_aperture` -- Uses a static aperture for all cadences based on the median image over all
   cadences
#. `compute_centroids_dynamic_aperture` -- Uses a dynamic aperture where a new aperture is computed for each cadence

You can then generate an animation of the computed centroids as the below example will demonstrate: 

.. code-block:: python

    >>>f = fits.open('.../EDR3/pdart-edr3/pdart-edr3-asteroid49016-s10-mtpf.fits')
    >>>mtpf = MovingTargetPixelFile(f, quality_bitmask="hardest")
    >>>centroids = MovingCentroids(mtpf)
    >>>centroidsMatrix = centroids.compute_centroids_dynamic_aperture(aper_mask_threshold=3)
    >>>centroids.mtpf.animate(aperture_mask=centroids.aper, centroidsMatrix=centroidsMatrix, fps=2, step=1)

..
    .. image:: https://raw.githubusercontent.com/lightkurve/lightkurve/main/docs/source/_static/images/lightkurve-teaser.gif

.. image:: https://raw.githubusercontent.com/jcsmithhere/tess-astrometry/main/docs/images/tgt_49016_dynamic_aperture_example_short.gif


You can also compare the measured astrometry to that given by JPL Horizons:

.. code-block:: python

    >>>centroids.detrend_centroids_expected_trend(plot=True, extra_title='Target {}'.format(centroids.targetid));

.. image:: https://raw.githubusercontent.com/jcsmithhere/tess-astrometry/main/docs/images/example_compare_with_JPL.png
    
