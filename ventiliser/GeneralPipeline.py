#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:12:50 2020

@author: David Chong Tian Wei
"""

from ventiliser import preprocess as pp
from ventiliser.StateMapper import StateMapper
from ventiliser.PhaseLabeller import PhaseLabeller
import json
import datetime

class GeneralPipeline:
    """ 
    Utility class to help tie the different parts of the package together into an easy to use pipeline
    
    Attributes
    ----------
    data : Pandas Dataframe
        The record to be analyzed. Populated using load_data
    mapper : StateMapper object
        Used for performing the state mapping phase
    labeller : PhaseLabeller object
        Used for peforming the segmentation and sub-phase labelling steps. Also will contain breaths after processing.
    config : Dictionary
        Used to store configuration details for logging
    configured : Boolean
        Whether configure has been called on the object
    data_loaded : Boolean
        Whether load_data has been called on the object
    """
    def __init__(self):
        """
        Initialises the pipeline with placeholder objects
        """
        self.data = None
        self.mapper = StateMapper()
        self.labeller = PhaseLabeller()
        self.config = {}
        self.configured = False
        self.data_loaded = False

    def load_data(self, path, cols):
        """
        Loads the data specified by path and cols and performs linear interpolation with window average baseline correction
        
        Parameters
        ----------
        path : string
            Path to the data file
        cols : array like of int
            Columns in the data file corresponding to time, pressure, and flow respectively
        
        Returns
        -------
        None
        """
        if not self.configured:
            print("Please configure the pipeline first")
            return
        self.data = pp.pre_process_ventilation_data(path, cols)
        if self.config["correction_window"] is not None:
            pp.correct_baseline(self.data.iloc[:,2], self.config["correction_window"])
        self.data.iloc[:,2] = self.flow_unit_converter(self.data.iloc[:,2])
        self.config["input_file"] = path
        self.data_loaded = True
        
    def configure(self,correction_window=None, flow_unit_converter=lambda x:x,
                  freq=100, peep=5.5, flow_thresh=0.1, t_len=0.03, f_base=0,
                  leak_perc_thresh=0.66, permit_double_cycling=False,
                  insp_hold_length=0.5, exp_hold_length=0.05):
        """ 
        Overall coniguration for the pipeline. Please call before process and load data
        
        Parameters
        ----------
        correction_window : int, optional
            Size of the window to perform baseline correction by centering on average. Defaults to None (no correction).
        flow_unit_converter : f: real -> real, optional
            Function to convert units of flow and flow_threshold to desired units to be displayed. Defaults to the identity function.
        freq : int, optional
            Sampling rate of the sample being analyzed. Defaults to 100
        peep : real, optional
            The value which will be considered baseline pressure. Defaults to 5.5
        flow_thresh : real, optional
            The minimum threshold that flow must cross to be considered a new breath. Defaults to 0.1
        t_len : real, optional
            Length of the window in seconds to perform state mapping. Defaults to 0.03
        f_base : real, optional
            Value for flow to be considered no_flow. Defaults to 0
        leak_perc_thresh : real, optional
            Maximum percentage difference between inspiratory and expiratory volume for a breath to be considered normal. Defaults to 66%.
        permit_double_cycling : boolean, optional
            Flag to decide whether to mergre double cycles. Defaults to false.
        insp_hold_length : real
            Maximum time in seconds from inspiration until an expiration is encountered, after which the breath is terminated. Defaults to 0.5
        exp_hold_length : real
            Maximum expiratory hold length between breaths to be considered double cycling. Defaults to 0.05s
        
        Returns
        -------
        None
        """
        self.config["correction_window"] = correction_window
        self.flow_unit_converter = flow_unit_converter
        self.config["freq"] = freq
        self.config["peep"] = peep
        self.config["flow_thresh"] = flow_unit_converter(flow_thresh)
        self.config["f_base"] = f_base
        self.config["t_len"] = t_len
        self.config["leak_perc_thresh"] = leak_perc_thresh
        self.config["permit_double_cycling"] = permit_double_cycling
        self.config["insp_hold_length"] = insp_hold_length
        self.config["exp_hold_length"] = exp_hold_length
        
        self.configured = True
        
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
        if not self.configured:
            print("Please configure the pipeline first")
            return
        if not self.data_loaded:
            print("Please load data first")
            return
        self.config["processing_start_time"] = datetime.datetime.now()
        
        self.mapper.configure(p_base=self.config["peep"],f_base=self.config["f_base"], 
                              f_thresh=self.config["flow_thresh"],freq=self.config["freq"],
                              t_len=self.config["t_len"])
        self.labeller.configure(freq=self.config["freq"],
                                hold_length=self.config["insp_hold_length"],
                                leak_perc_thresh=self.config["leak_perc_thresh"],
                                permit_double_cycling=self.config["permit_double_cycling"],
                                exp_hold_len=self.config["exp_hold_length"])
        self.mapper.process(self.data.iloc[:,1], self.data.iloc[:,2])
        self.labeller.process(self.mapper.p_labels, self.mapper.f_labels,
                              self.data.iloc[:,1], self.data.iloc[:,2])
        self.config["processing_end_time"] = datetime.datetime.now()
        self.config["time_elapsed"] = str(self.config["processing_end_time"] - self.config["processing_start_time"])
        stem = ".".join(self.config["input_file"].split(".")[:-1])
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
                