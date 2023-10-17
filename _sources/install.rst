Installation
============

.. _installation_deps:

Dependencies
------------

dSGP4 has the following Python dependencies:

* `NumPy <https://numpy.org/>`__ (**mandatory**),

Packages
--------

conda
^^^^^

coming soon

pip
^^^

coming soon

Installation from source
------------------------


Using ``git``:

.. code-block:: console

   $ git clone https://github.com/esa/dSGP4

We follow the usual PR-based development workflow, thus dSGP4's ``main``
branch is normally kept in a working state.

After downloading and/or unpacking heyoka.py's
source code, go to heyoka.py's
source tree, create a ``build`` directory and ``cd`` into it. E.g.,
on a Unix-like system:

.. code-block:: console

   $ cd /path/to/heyoka.py
   $ mkdir build
   $ cd build

Once you are in the ``build`` directory, you must configure your build
using ``cmake``. There are various useful CMake variables you can set,
such as:

* ``CMAKE_BUILD_TYPE``: the build type (``Release``, ``Debug``, etc.),
  defaults to ``Release``.
* ``CMAKE_PREFIX_PATH``: additional paths that will be searched by CMake
  when looking for dependencies.
* ``HEYOKA_PY_INSTALL_PATH``: the path into which the heyoka.py module
  will be installed. If left empty (the default), heyoka.py will be installed
  in the global modules directory of your Python installation.
* ``HEYOKA_PY_ENABLE_IPO``: set this flag to ``ON`` to compile heyoka.py
  with link-time optimisations. Requires compiler support,
  defaults to ``OFF``.

Please consult `CMake's documentation <https://cmake.org/cmake/help/latest/>`_
for more details about CMake's variables and options.

The ``HEYOKA_PY_INSTALL_PATH`` option is particularly important. If you
want to install heyoka.py locally instead of globally (which is in general
a good idea), you can set this variable to the output of
``python -m site --user-site``.

After configuring the build with CMake, we can then proceed to actually
building heyoka.py:

.. code-block:: console

   $ cmake --build .

Finally, we can install heyoka.py with the command:

.. code-block:: console

   $ cmake  --build . --target install

Verifying the installation
--------------------------

You can verify that dSGP4 was successfully compiled and
installed by running the tests. To do so, you must first install the
optional dependencies (see :ref:`installation_deps`), then run the

.. code-block:: bash

   $ pytest

If this command executes without any error, then
your dSGP4 installation is ready for use.

Getting help
------------

If you run into troubles installing heyoka.py, please do not hesitate
to contact us by opening an issue report on `github <https://github.com/esa/dSGP4/issues>`__.