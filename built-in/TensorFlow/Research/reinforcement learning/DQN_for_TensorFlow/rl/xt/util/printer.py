#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Printer utils
"""

import sys


def print_immediately(to_str):
    """print some string immediately"""
    print(to_str)
    sys.stdout.flush()
