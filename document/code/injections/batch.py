#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import sys
from subprocess import Popen

jobs = []
for i in range(int(sys.argv[1])):
    jobs.append(Popen(["python", "injections.py"]))

[j.communicate() for j in jobs]
