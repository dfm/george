# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "izip", "imap",
    "xrange", "basestring",
    "iteritems"
]

import sys

if sys.version_info[0] < 3:
    from itertools import izip, imap

    xrange = xrange
    basestring = basestring

    def iteritems(d):
        return d.iteritems()

else:
    izip = zip
    imap = map
    xrange = range
    basestring = str

    def iteritems(d):
        return d.items()
