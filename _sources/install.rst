Installation
============

.. _installation_deps:

Dependencies
------------

dSGP4 has the following Python dependencies:



Packages
--------

conda
^^^^^

First, we add `conda-forge` channel to the channels:
.. code-block:: console
   
   $ conda config --add channels conda-forge
   $ conda config --set channel_priority strict

Now, we can install `dsgp4` either through `conda`:
.. code-block:: console
   
   $ conda install dsgp4

or `mamba`:

.. code-block:: console
   
   $ mamba install dsgp4



pip
^^^

`dsgp4` is available on [Pypi](https://pypi.org/project/dsgp4/). You can install it via `pip` as:

.. code-block:: console
   
   $ pip install dsgp4

Installation from source
------------------------


Using ``git``:

.. code-block:: console

   $ git clone https://github.com/esa/dSGP4

We follow the usual PR-based development workflow, thus dSGP4's ``master``
branch is normally kept in a working state.

Verifying the installation
--------------------------

You can verify that dSGP4 was successfully compiled and
installed by running the tests. To do so, you must first install the
optional dependencies.

.. code-block:: bash

   $ pytest

If this command executes without any error, then
your dSGP4 installation is ready for use.

Getting help
------------

If you run into troubles installing dsgp4, please do not hesitate
to contact us by opening an issue report on `github <https://github.com/esa/dSGP4/issues>`__.
