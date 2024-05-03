.. _dev_testing:

Testing
=======

pytest
------

For rapid testing, simply run `pytest`, but exclude large tests.

.. code::

   pytest -vv -m "not large"


tox
---

For testing all Python environments, use `tox`.

.. code::

   tox

Note that this randomizes the order of tests but skips "large" tests. To modify this behavior, `tox` has to skip the optional *posargs*.

.. code::

   tox -- test

To test specific environments, use the `-e` flag.

.. code::

   tox -e py39-torch1110
