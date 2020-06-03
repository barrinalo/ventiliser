#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:12:50 2020

@author: David Chong Tian Wei
"""

from ventiliser import preprocess as pp
from ventiliser.StateMapper import StateMapper
from ventiliser.PhaseLabeller import PhaseLabeller

class GeneralPipeline:
    
    def __init__(self):
        self.data = None
        self.mapper = StateMapper()
        self.labeller = PhaseLabeller()
        self.configured = False
        self.data_loaded = False

    def load_data(self, path, cols):
        """Loads the data specified by path and cols and performs linear interpolation with window average baseline correction
        
        :param path: Path to data file
        :type path: string
        :param cols: Columns in the data file corresponding to time, pressure, and flow respectively
        :type cols: Array like of integer
        :returns: None
        :rtype: None
        """
        if not self.configured:
            print("Please configure the pipeline first")
            return
        self.data = pp.pre_process_ventilation_data(path, cols)
        if self.correction_window is not None:
            pp.correct_baseline(self.data.iloc[:,2], self.correction_window)
        self.data.iloc[:,2] = self.flow_unit_converter(self.data.iloc[:,2])
        self.data_loaded = True
        
    def configure(self,correction_window=None, flow_unit_converter=lambda x:x,
                  freq=100, peep=5.5, flow_thresh=0.1, w_len=3, f_base=0,
                  leak_perc_thresh=0.66, permit_double_cycling=False,
                  insp_hold_length=0.5, exp_hold_length=0.05):
        """ Overall coniguration for the pipeline. Please call before process and load data
        
        :param correction_window: Size of the window to perform baseline correction by centering on average
        :type correction_window: None or positive integer
        :param flow_unit_converter: Function to convert units of flow and flow_threshold to desired units to be displayed
        :type flow_unit_converter: f: R->R
        :param freq: Sampling rate of the sample being analyzed
        :type freq: integer
        :param peep: The value which will be considered baseline pressure
        :type peep: real
        :param flow_thresh: The minimum threshold that flow must cross to be considered a new breath
        :type flow_thresh: real
        :param w_len: Length of the window in data points to perform state mapping
        :type w_len: integer
        :param f_base: Value for flow to be considered no_flow
        :type f_base: real
        :param leak_perc_thresh: Maximum percentage difference between inspiratory and expiratory volume for a breath to be considered normal
        :type leak_perc_thresh: real
        :param permit_double_cycling: Whether double cycles will be merged into single breath
        :type permit_double_cycling: boolean
        :param insp_hold_length: Maximum time in seconds from inspiration until an expiration is encountered, after which the breath is terminated
        :type insp_hold_length: real
        :param exp_hold_length: Maximum expiratory hold length between breaths to be considered double cycling
        :type exp_hold_length: real
        
        :returns: None
        :rtype: None
        """
        self.correction_window = correction_window
        self.flow_unit_converter = flow_unit_converter
        self.freq = freq
        self.peep = peep
        self.flow_thresh = flow_unit_converter(flow_thresh)
        self.f_base = f_base
        self.w_len = w_len
        self.t_len = 1 / freq * w_len
        self.leak_perc_thresh = leak_perc_thresh
        self.permit_double_cycling = permit_double_cycling
        self.insp_hold_length = insp_hold_length
        self.exp_hold_length = exp_hold_length
        
        self.configured = True
        
    def process(self):
        """Processes the data after configuration and loading of data
        """
        if not self.configured:
            print("Please configure the pipeline first")
            return
        if not self.data_loaded:
            print("Please load data first")
            return
        self.mapper.configure(p_base=self.peep,f_base=self.f_base, 
                              f_thresh=self.flow_thresh,freq=self.freq,
                              t_len=self.t_len)
        self.labeller.configure(w_len=self.w_len,freq=self.freq,
                                hold_length=self.insp_hold_length,
                                leak_perc_thresh=self.leak_perc_thresh,
                                permit_double_cycling=self.permit_double_cycling,
                                exp_hold_len=self.exp_hold_length)
        self.mapper.process(self.data.iloc[:,1], self.data.iloc[:,2])
        self.labeller.process(self.mapper.p_labels, self.mapper.f_labels,
                              self.data.iloc[:,1], self.data.iloc[:,2])