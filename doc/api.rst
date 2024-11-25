.. _api:

API
#######

$\partial$SGP4 API

.. currentmodule:: dsgp4

.. toctree::
   :maxdepth: 2

   _autosummary/dsgp4
   _autosummary/dsgp4.mldsgp4

.. autosummary::
   :toctree: _autosummary/
   :recursive:

   plot.plot_orbit
   plot.plot_tles
   tle.compute_checksum
   tle.read_satellite_catalog_number
   tle.load_from_lines
   tle.load_from_data
   tle.load
   tle.TLE
   tle.TLE.copy
   tle.TLE.perigee_alt
   tle.TLE.apogee_alt
   tle.TLE.set_time
   tle.TLE.update
   util.get_gravity_constants
   util.propagate_batch
   util.propagate
   util.initialize_tle
   util.from_year_day_to_date
   util.gstime
   util.clone_w_grad
   util.jday
   util.invjday
   util.days2mdhms
   util.from_string_to_datetime
   util.from_mjd_to_epoch_days_after_1_jan
   util.from_mjd_to_datetime
   util.from_jd_to_datetime
   util.get_non_empty_lines
   util.from_datetime_to_fractional_day
   util.from_datetime_to_mjd
   util.from_datetime_to_jd
   util.from_cartesian_to_tle_elements
   util.from_cartesian_to_keplerian
   util.from_cartesian_to_keplerian_torch
   sgp4
   sgp4_batched
   sgp4init.sgp4init
   sgp4init_batch.sgp4init_batch
   sgp4init_batch.initl_batch
   initl
   newton_method
   sgp4init
   sgp4init_batch

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
 
   mldsgp4.mldsgp4
