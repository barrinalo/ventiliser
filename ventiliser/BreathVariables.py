#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:56:04 2020

@author: David Chong Tian Wei
"""

class BreathVariables:
    """
    Structure to hold measurements of each breath.
    
    Attributes
    ----------
    breath_number : int
        Index for the breath with respect to the analysis that was run
    breath_start : int
        The index in the input waveform data corresponding to the start of the current breath
    breath_end : int
        The index in the input waveform data corresponding to the end of the current breath
    pressure_rise_start : int
        The index in the input waveform data corresponding to the start of the pressure rise phase of the current breath
    pip_start : int
        The index in the input waveform data corresponding to the start of the peak inflation pressure phase of the current breath
    pressure_drop_start : int
        The index in the input waveform data corresponding to the start of the pressure drop phase of the current breath
    peep_start : int
        The index in the input waveform data corresponding to the start of the positive end expiratory pressure phase of the current breath
    inspiration_initiation_start : int
        The index in the input waveform data corresponding to the start of the inspiration initiation phase of the current breath
    inspiratory_hold_start : int
        The index in the input waveform data corresponding to the start of the inspiratory hold phase of the current breath
    peak_inspiratory_flow_start : int
        The index in the input waveform data corresponding to the start of the peak inspiratory flow phase of the current breath
    inspiration_termination_start : int
        The index in the input waveform data corresponding to the start of the inspiration termination phase of the current breath  
    inspiratory_hold_start : int
        The index in the input waveform data corresponding to the start of the inspiratory hold phase of the current breath
    expiration_initiation_start : int
        The index in the input waveform data corresponding to the start of the expiration initiation phase of the current breath
    peak_expiratory_flow_start : int
        The index in the input waveform data corresponding to the start of the peak expiratory flow phase of the current breath
    expiration_termination_start : int
        The index in the input waveform data corresponding to the start of the expiration termination phase of the current breath
    expiratory_hold_start : int
        The index in the input waveform data corresponding to the start of the expiratory hold phase of the current breath
    pressure_rise_length : int
        The length of the pressure rise phase of the current breath in terms of number of time units
    pip_length : int
        The length of the peak inspiratory pressure phase of the current breath in terms of number of time units
    peep_length : int
        The length of the positive end expiratory pressure phase of the current breath in terms of number of time units
    inspiration_initiation_length : int
        The length of the inspiration initiation phase of the current breath in terms of number of time units
    peak_inpiratory_flow_length : int
        The length of the peak inspiratory flow phase of the current breath in terms of number of time units
    inspiration_termination_length : int
        The length of the inspiration termination phase of the current breath in terms of number of time units
    inspiratory_hold_length : int
        The length of the inspiratory hold phase of the current breath in terms of number of time units
    expiration_initiation_length : int
        The length of the expiration initiation phase of the current breath in terms of number of time units
    peak_expiratory_flow_length : int
        The length of the peak expiratory flow phase of the current breath in terms of number of time units
    expiration_termination_length : int
        The length of the expiration termination phase of the current breath in terms of number of time units
    expiratory_hold_length : int
        The length of the expiratory hold phase of the current breath in terms of number of time units
    pip_to_no_flow_length : int
        The length of the period from the start of peak inspiratory pressure phase to the start of inspiratory hold phase in terms of number of time units
    peep_to_no_flow : int
        The length of the period from the start of the positive end expiratory pressure phase to the start of expiratory hold phase in terms of the number of time units
    lung_inflation_length : int
        The length of the period from the start of inspiration initiation phase to the start of the inspiratory hold phase in terms of number of time units
    lung_deflation_length : int
        The length of the period from the start of expiration initiation phase to the start of the expiratory hold phase in terms of number of time units
    total_inspiratory_length : int
        The length of the period from the start of inspiration initiation phase to the start of the expiration initiation in terms of number of time units
    total_expiratory_length : int
        The length of the period from the start of expiratory initiation phase to the end of the breath in terms of number of time units
    inspiratory_volume : real
        The signed volume of inspiration calculated by summing the positive flow values in the current breath
    expiratory_volume : real
        The signed volume of expiration calculated by summing the negative flow values in the current breath
    max_inspiratory_flow : real
        The most positive flow value in the current breath
    max_expiratory_flow : real
        The most negative flow value in the current breath
    max_pressure : real
        The largest pressure value in the current breath
    min_pressure : real
        The smallest pressure value in the current breath
    pressure_flow_correlation : real
        Pearson correlation coefficient of the pressure and flow for the current breath
    
    Methods
    --------
    valid()
        Checks if the indices of the key points are physiologically valid in terms of order
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
        """
        Checks if the indices of the key points are physiologically valid in terms of order
        
        Returns
        -------
        boolean
            Indicates if the indices of the key points are physiologically valid in terms of order
        """
        return (self.pressure_rise_start <= self.pip_start <= self.pressure_drop_start <= self.peep_start) and ((self.pressure_rise_length + self.pip_length + self.pressure_drop_start + self.peep_length) == (self.breath_end - self.breath_start)) and (self.inspiration_initiation_start <= self.peak_inspiratory_flow_start <= self.inspiration_termination_start <= self.inspiratory_hold_start <= self.expiration_initiation_start <= self.peak_expiratory_flow_start <= self.expiration_termination_start <= self.expiratory_hold_start) and ((self.inspiration_initiation_length + self.peak_inspiratory_flow_length + self.inspiration_termination_length + self.inspiratory_hold_length + self.expiration_initiation_length + self.peak_expiratory_flow_length + self.expiration_termination_length + self.expiratory_hold_length) == (self.breath_end - self.breath_start))
