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
import json
import datetime

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
    config : Dictionary
        Used to store configuration details for logging
    """
    def __init__(self):
        self.tasks = None
        self.data = None
        self.mapper = StateMapper()
        self.labeller = PhaseLabeller()
        self.config = {}
        self.config["freq"] = 100
        self.config["f_base"] = 0
        self.config["t_len"] = 1 / 100 * 3
        self.config["leak_perc_thresh"] = 0.66
        self.config["permit_double_cycling"] = False
        self.config["insp_hold_length"] = 0.5
        self.config["exp_hold_length"] = 0.05
        
    def load_data(self, path, cols, correction_window=None, flow_unit_converter=lambda x : x):
        """
        Loads the data specified by path and cols and performs linear interpolation with window average baseline correction
        
        Parameters
        ----------
        path : string
            Path to the data file
        cols : array like of int
            Columns in the data file corresponding to time, pressure, and flow respectively
        correction_window : int, optional
            Size of the window to perform baseline correction by centering on average. Defaults to None (no correction).
        flow_unit_converter : f: real -> real, optional
            Function to convert units of flow and flow_threshold to desired units to be displayed. Defaults to the identity function.
        
        Returns
        -------
        None
        """
        self.flow_unit_converter = flow_unit_converter
        self.config["correction_window"] = correction_window
        self.config["input_file"] = path
        self.data = pp.pre_process_ventilation_data(path, cols)
        if correction_window is not None:
            pp.correct_baseline(self.data.iloc[:,2], correction_window)
        self.data.iloc[:,2] = flow_unit_converter(self.data.iloc[:,2])
        self.flow_threshold = flow_unit_converter(0.1)
        self.config["flow_thresh"] = self.flow_threshold
        
    def load_mapper_settings(self, path):
        """
        Loads the settings file to look for changes in PEEP setting
        
        Parameters
        ----------
        path : string
            Path to the settings file
            
        Returns
        -------
        None
        """
        settings = pd.read_csv(path)
        self.tasks = list(settings.loc[settings["Name"]=="PEEP"].apply(lambda x: (x["Time [ms]"], x["Value New"]), axis=1))
        self.config["peeps"] = self.tasks
    
    def _map_states(self):
        """
        Runs the mapper on the data with provided settings. Splits up the data into periods when differente levels of PEEP are used
        
        Returns
        -------
        None
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
            
    def process(self, log=True, output_files=True):
        """
        Processes the data after configuration and loading of data
        
        Parameters
        ----------
        log : boolean, optional
            Flag to decide whether to create output logs for the analysis. Defaults to True
        output_files : boolean, optional
            Flag to decide whether to create output files. Defaults to True
        
        Returns
        -------
        None
        """
        # Start log
        self.config["processing_start_time"] = datetime.datetime.now()
        # Process
        self._map_states()
        self.labeller.process(self.mapper.p_labels, self.mapper.f_labels, self.data.iloc[:,1], self.data.iloc[:,2])
        # Finish logging
        self.config["processing_end_time"] = datetime.datetime.now()
        self.config["time_elapsed_ms"] = str(self.config["processing_end_time"] - self.config["processing_start_time"])
        stem = ".".join(self.config["input_file"].split(".")[:-1])
        # Output files if flags set
        if output_files:
            breaths_raw = self.labeller.get_breaths_raw()
            breaths_raw["max_expiratory_flow"] = breaths_raw["max_expiratory_flow"].apply(lambda x : x / self.flow_unit_converter(1))
            breaths_raw["max_inspiratory_flow"] = breaths_raw["max_inspiratory_flow"].apply(lambda x : x / self.flow_unit_converter(1))
            breaths_raw.to_csv(stem + "_predicted_Breaths_Raw.csv", index=False)
            breaths = self.labeller.get_breaths()
            breaths["max_expiratory_flow"] = breaths["max_expiratory_flow"].apply(lambda x : x / self.flow_unit_converter(1))
            breaths["max_inspiratory_flow"] = breaths["max_inspiratory_flow"].apply(lambda x : x / self.flow_unit_converter(1))
            breaths.to_csv(stem + "_predicted_Breaths_ms.csv", index=False)
            self.mapper.get_labels().to_csv(stem + "_predicted_Pressure_And_Flow_States.csv", index=False)
            self.labeller.get_breath_annotations(self.data.shape[0]).to_csv(stem + "_predicted_Breaths_Annotations.csv", index=False)
            self.config["output_files"] = [stem + "_predicted_Breaths_Raw.csv",
                                           stem + "_predicted_Breaths_ms.csv",
                                           stem + "_predicted_Pressure_And_Flow_States.csv",
                                           stem + "_predicted_Breaths_Annotations.csv"]
        if log:
            self.config["processing_start_time"] = str(self.config["processing_start_time"])
            self.config["processing_end_time"] = str(self.config["processing_end_time"])
            f = open(stem + "_run_config.json","w")
            f.write(json.dumps(self.config))
            f.close()
