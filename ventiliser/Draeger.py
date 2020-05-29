#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:49:27 2020

@author: David Chong Tian Wei
"""

import pandas as pd
from ventiliser import preprocess as pp
from ventiliser.StateMapper import StateMapper
from ventiliser.PhaseLabeller import PhaseLabeller

class Draeger:
    """ Class to process Draeger ventilator data
    
    Attributes
    ----------
    tasks : Array of 2-tuple (integer, real)
        Populated when load_mapper_settings is called with 2-tuples of time [ms] and PEEP set at that time
    data : Pandas Dataframe
        The record to be analyzed. Populated using load_data
    mapper : StateMapper object
        Used for performing the state mapping phase
    labeller : PhaseLabeller object
        Used for peforming the segmentation and sub-phase labelling steps. Also will contain breaths after processing.
        
    Methods
    -------
    load_data(path, cols, correction_window)
        Loads a record and pre-processes it with linear interpolation and baseline correction
    load_mapper_settings(path)
        Loads the settings file and determines how to split the data up into periods to account for varying ventilator settings
    process()
        Runs the mapper and then the labeller process methods
    """
    def __init__(self):
        self.tasks = None
        self.data = None
        self.mapper = StateMapper()
        self.labeller = PhaseLabeller()
        
    def load_data(self, path, cols, correction_window=None, flow_unit_converter=lambda x : x):
        """Loads the data specified by path and cols and performs linear interpolation with window average baseline correction
        
        :param path: Path to data file
        :type path: string
        :param cols: Columns in the data file corresponding to time, pressure, and flow respectively
        :type cols: Array like of integer
        :param correction_window: Size of window to perform baseline correction
        :type correction_window: integer
        :returns: None
        :rtype: None
        """
        self.data = pp.pre_process_ventilation_data(path, cols)
        if correction_window is not None:
            pp.correct_baseline(self.data.iloc[:,2], correction_window)
        self.data.iloc[:,2] = flow_unit_converter(self.data.iloc[:,2])
        self.flow_threshold = flow_unit_converter(0.1)
        
    def load_mapper_settings(self, path):
        """Loads the settings file to look for changes in PEEP setting
        
        :param path: Path to settings file
        :type path: string
        :returns: None
        :rtype: None
        """
        settings = pd.read_csv(path)
        self.tasks = list(settings.loc[settings["Name"]=="PEEP"].apply(lambda x: (x["Time [ms]"], x["Value New"]), axis=1))
    
    def _map_states(self):
        """Runs the mapper on the data with provided settings. Splits up the data into periods when differente levels of PEEP are used
        
        :returns: None
        :rtype: None
        """
        if self.data is None:
            print("Error: No data has been loaded")
            return
        if self.tasks is None:
            print("Warning: No settings file has been loaded, proceeding with default values")
            self.mapper.configure(f_thresh=self.flow_threshold)
            self.mapper.process(self.data.iloc[:,1], self.data.iloc[:,2])
        else:
            prev_t = self.data.iloc[:, 0].min()
            prev_p = 5.5 # Default 5.5 peep
            for t, p in self.tasks:
                period = (self.data.iloc[:,0] >= prev_t) & (self.data.iloc[:,0] < t)
                if self.data.loc[period].shape[0] > 0:
                    self.mapper.configure(p_base=prev_p, f_thresh=self.flow_threshold)
                    self.mapper.process(self.data.loc[period].iloc[:,1], self.data.loc[period].iloc[:,2])
                prev_t = t
                prev_p = p
            period = (self.data.iloc[:,0] >= prev_t) & (self.data.iloc[:,0] < self.data.iloc[:,0].max()+1)
            if self.data.loc[period].shape[0] > 0:
                self.mapper.configure(p_base=prev_p, f_thresh=self.flow_threshold)
                self.mapper.process(self.data.loc[period].iloc[:,1], self.data.loc[period].iloc[:,2])
            
    def process(self):
        self._map_states()
        self.labeller.process(self.mapper.p_labels, self.mapper.f_labels, self.data.iloc[:,1], self.data.iloc[:,2])
