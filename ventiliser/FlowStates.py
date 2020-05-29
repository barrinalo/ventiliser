#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:09:31 2020

@author: David Chong Tian Wei
"""
import enum

class FlowStates(enum.Enum):
    """Enumeration for flow states"""
    no_flow = 0
    inspiration_initiation = 1
    peak_inspiratory_flow = 2
    inspiration_termination = 3
    expiration_initiation = 4
    peak_expiratory_flow = 5
    expiration_termination = 6