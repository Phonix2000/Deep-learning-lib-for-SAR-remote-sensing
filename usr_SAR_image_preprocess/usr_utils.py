#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import progressbar
from scipy import linalg


def usr_upbounded(x,upper):
    retain_mask = np.asarray([x<= upper], dtype= x.dtype)
    bounded_mask = np.asarray([x> upper], dtype= x.dtype) * upper
    bounded_x = x*retain_mask + bounded_mask
    return bounded_x

def usr_lowbounded(x,lower):
    retain_mask = np.asarray([x>= lower], dtype= x.dtype)
    bounded_mask = np.asarray([x< lower], dtype= x.dtype) *lower
    bounded_x = x*retain_mask + bounded_mask
    return bounded_x

def usr_up_low_bounded(x,minval,maxval):
    return usr_lowbounded(usr_upbounded(x,maxval),minval)
