#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:09:54 2020

@author: David Chong Tian Wei
"""
import enum

class PressureStates(enum.Enum):
    """
    Enumeration for pressure states
    
    Attributes
    ----------
    peep : int
        0
    pressure_rise : int
        1
    pip : int
        2
    pressure_drop : int
        3
    """
    peep = 0
    pressure_rise = 1
    pip = 2
    pressure_drop = 3
    
