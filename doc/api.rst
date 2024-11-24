.. _api:

API
====

$\partial$SGP4 API

.. autosummary::
   :toctree: _autosummary
   :recursive:

   dsgp4
   dsgp4.mldsgp4.mldsgp4
   dsgp4.plot.plot_orbit
   dsgp4.plot.plot_tles
   dsgp4.tle.compute_checksum
   dsgp4.tle.read_satellite_catalog_number
   dsgp4.tle.load_from_lines
   dsgp4.tle.load_from_data
   dsgp4.tle.load
   dsgp4.tle.TLE
   dsgp4.tle.TLE.copy
   dsgp4.tle.TLE.perigee_alt
   dsgp4.tle.TLE.apogee_alt
   dsgp4.tle.TLE.set_time
   dsgp4.tle.TLE.update
   dsgp4.util.get_gravity_constants
   dsgp4.util.propagate_batch
   dsgp4.util.propagate
   dsgp4.util.initialize_tle
   dsgp4.util.from_year_day_to_date
   dsgp4.util.gstime
   dsgp4.util.clone_w_grad
   dsgp4.util.jday
   dsgp4.util.invjday
   dsgp4.util.days2mdhms
   dsgp4.util.from_string_to_datetime
   dsgp4.util.from_mjd_to_epoch_days_after_1_jan
   dsgp4.util.from_mjd_to_datetime
   dsgp4.util.from_jd_to_datetime
   dsgp4.util.get_non_empty_lines
   dsgp4.util.from_datetime_to_fractional_day
   dsgp4.util.from_datetime_to_mjd
   dsgp4.util.from_datetime_to_jd
   dsgp4.util.from_cartesian_to_tle_elements
   dsgp4.util.from_cartesian_to_keplerian
   dsgp4.util.from_cartesian_to_keplerian_torch
   dsgp4.sgp4
   dsgp4.sgp4_batched
   dsgp4.sgp4init.sgp4init
   dsgp4.sgp4init_batch.sgp4init_batch
   dsgp4.sgp4init_batch.initl_batch