#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:09:54 2020

@author: David Chong Tian Wei
"""
import enum

class PressureStates(enum.Enum):
    """Enumeration for pressure states"""
    peep = 0
    pressure_rise = 1
    pip = 2
    pressure_drop = 3
    
