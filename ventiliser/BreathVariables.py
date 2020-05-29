#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:56:04 2020

@author: David Chong Tian Wei
"""

class BreathVariables:
    """Model for variables to record for each breath
    
    Attributes
    ----------
    start
    """
    def __init__(self):
        # Points in time
        self.breath_number = None
        self.breath_start = None
        self.breath_end = None
        self.pressure_rise_start = None
        self.pip_start = None
        self.pressure_drop_start = None
        self.peep_start = None
        self.inspiration_initiation_start = None
        self.peak_inspiratory_flow_start = None
        self.inspiration_termination_start = None
        self.inspiratory_hold_start = None
        self.expiration_initiation_start = None
        self.peak_expiratory_flow_start = None
        self.expiration_termination_start = None
        self.expiratory_hold_start = None
        # Length of phases
        self.pressure_rise_length = None
        self.pip_length = None
        self.pressure_drop_length = None
        self.peep_length = None
        self.inspiration_initiation_length = None
        self.peak_inspiratory_flow_length = None
        self.inspiration_termination_length = None
        self.inspiratory_hold_length = None
        self.expiration_initiation_length = None
        self.peak_expiratory_flow_length = None
        self.expiration_termination_length = None
        self.expiratory_hold_length = None
        self.pip_to_no_flow_length = None
        self.peep_to_no_flow_length = None
        self.lung_inflation_length = None
        self.total_inspiratory_length = None
        self.lung_deflation_length = None
        self.total_expiratory_length = None
        # Volumes
        self.inspiratory_volume = None
        self.expiratory_volume = None
        # Extreme values
        self.max_inspiratory_flow = None
        self.max_expiratory_flow = None
        self.max_pressure = None
        self.min_pressure = None
        # Misc
        self.pressure_flow_correlation = None
    
    def valid(self):
        return (self.pressure_rise_start <= self.pip_start <= self.pressure_drop_start <= self.peep_start) and ((self.pressure_rise_length + self.pip_length + self.pressure_drop_start + self.peep_length) == (self.breath_end - self.breath_start)) and (self.inspiration_initiation_start <= self.peak_inspiratory_flow_start <= self.inspiration_termination_start <= self.inspiratory_hold_start <= self.expiration_initiation_start <= self.peak_expiratory_flow_start <= self.expiration_termination_start <= self.expiratory_hold_start) and ((self.inspiration_initiation_length + self.peak_inspiratory_flow_length + self.inspiration_termination_length + self.inspiratory_hold_length + self.expiration_initiation_length + self.peak_expiratory_flow_length + self.expiration_termination_length + self.expiratory_hold_length) == (self.breath_end - self.breath_start))
