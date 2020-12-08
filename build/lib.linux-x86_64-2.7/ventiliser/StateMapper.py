#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:44:47 2020

@author: David Chong Tian Wei
"""
import math
import numpy as np
import pandas as pd
from atpbar import register_reporter, find_reporter, atpbar
from ventiliser.FlowStates import FlowStates as fs
from ventiliser.PressureStates import PressureStates as ps
from multiprocessing import Process, Queue, cpu_count

class StateMapper:
    """Class for mapping pressure and flow time series to enumerated states.
    
    Attributes
    ----------
    p_base : real
        Baseline pressure to be used in processing. Usually can be found from ventilator settings
    f_base : real
        Baseline flow to be used in processing. Is usually set to 0
    f_thresh : real
        Threshold which flow must be larger than in order to be considered a moving state
    freq : real
        Sampling rate for the record to be processed in Hz
    w_len : integer
        Number of datapoints to be used in a window calculation
    p_labels : Numpy array of PressureStates enum
        Contains the calculated pressure state labels after process is called
    f_labels : Numpy array of FlowStates enum
        Contains the calculated flow state labels after process is called
    """
    
    def __init__(self):
        self.p_labels = np.array([])
        self.f_labels = np.array([])
        self.configure();
    
    def configure(self, p_base=5.5, f_base=0, f_thresh=0.1, freq=100, t_len=0.03):
        """ 
        Sets processing constants. To be called before process
        
        Parameters
        ----------
        p_base : real, optional
            Baseline pressure / PEEP set on the ventilator. Defaults to 5.5
        f_base : real, optional
            Baseline flow which is usually 0. Defaults to 0
        f_thresh : real, optional
            Threshold for the standard deviation of a window to be considered non-stationary. Defaults to 0.1
        freq : int, optional
            Sampling rate of the input to be processed. Defaults to 100
        t_len : real, optional
            Length of time in seconds of the desired window to use. Defaults to 0.03s
        
        Returns
        -------
        None
        """
        self.p_base = p_base
        self.f_base = f_base
        self.f_thresh = f_thresh
        self.freq = freq
        self.w_len = math.ceil(freq * t_len)
        if self.w_len < 3:
            print("Warning: calculated window length is less than 3, average and standard deviation calculations may be unhelpful, consider increasing t_len")
    
    def get_labels(self):
        """
        Returns the calculated labels
        
        Returns
        -------
        Pandas dataframe
            Dataframe containing the calculated pressure and flow states as integers
        """
        return pd.DataFrame({"Pressure_States" : [x.value for x in self.p_labels], "Flow_States" : [x.value for x in self.f_labels]})
            
    def process_pressures(self, pressures, p_0=ps.peep, con=None, reporter=None):
        """
        Maps data points from pressure to enumerated states
        
        Parameters
        ----------
        pressures : array like of real
            Pressure data points
        p_0 : PressureStates enum, optional
            The initial pressure state the program assumes it is in. Defaults to peep
        con : Queue object, optional
            The queue object to use for transferring data back to main cpu if multiprocessing is used. Defaults to None
        reporter : reporter object, optional
            The reporter object used to update the main cpu on the progress of the analysis. Defaults to None
        
        Returns
        -------
        None
        """
        if len(pressures) < self.w_len and con is None:
            con.put(np.array([]))
        elif len(pressures) < self.w_len:
            return
        
        output = np.array([ps(p_0)] * len(pressures))
        pressures = np.array(pressures).reshape(-1, self.w_len)
        pressure_means = np.mean(pressures, axis=1)
        pressure_mean_deltas = pressure_means[1:] - pressure_means[:-1]
        pressure_stds = np.std(pressures, axis=1)
        prev_label = ps(p_0)
        prev_pressure_hold = self.p_base
        if reporter != None:
            register_reporter(reporter)
        for i in atpbar(range(len(pressure_means)-1), name="Labelling pressure states"):
            if pressure_stds[i] < 0.1 * self.p_base:
                w_std_i = 0.05 * self.p_base
            else:
                w_std_i = pressure_stds[i]
            w_mean_delta = pressure_mean_deltas[i]
            w_mean_i_1 = pressure_means[i+1]
            curpos = (i+1) * self.w_len
            # If standard deviation is too small, set it to a minimum threshold
            if w_std_i < 0.1 * self.p_base:
                w_std_i = self.p_base * 0.05
            
            # Process stationary states
            if abs(w_mean_delta) < 2 * w_std_i:
                # Process for PIP
                if w_mean_i_1 > (self.p_base + prev_pressure_hold) / 2 + 2 * w_std_i:
                    if prev_label is not ps.pip:
                        output[curpos:curpos + self.w_len] = ps.pip
                        prev_pressure_hold = pressure_means[i+1]
                        prev_label = ps.pip
                    else:
                        output[curpos:curpos + self.w_len] = ps.pip
                        prev_label = ps.pip
                else:
                    # Process PEEP
                    if prev_label is not ps.peep:
                        output[curpos:curpos + self.w_len] = ps.peep
                        prev_pressure_hold = pressure_means[i+1]
                        prev_label = ps.peep
                    else:
                        output[curpos:curpos + self.w_len] = ps.peep
                        prev_label = ps.peep
            elif w_mean_delta > 0:
                # Process pressure rise
                output[curpos:curpos + self.w_len] = ps.pressure_rise
                prev_label = ps.pressure_rise
            else:
                # Process pressure drop
                output[curpos:curpos + self.w_len] = ps.pressure_drop
                prev_label = ps.pressure_drop
        if con is not None:
            con.put(output)
        else:
            self.p_labels = np.concatenate([self.p_labels, output])
            
    def process_flows(self, flows, f_0=fs.no_flow, con=None, reporter=None):
        """
        Maps data points from pressure to enumerated states
        
         Parameters
        ----------
        flows : array like of real
            Flow data points
        f_0 : FlowStates enum, optional
            The initial flow state the program assumes it is in. Defaults to no flow.
        con : Queue object, optional
            The queue object to use for transferring data back to main cpu if multiprocessing is used. Defaults to None
        reporter : reporter object, optional
            The reporter object used to update the main cpu on the progress of the analysis. Defaults to None
        
        Returns
        -------
        None
        """
        if len(flows) < self.w_len and con is None:
            con.put(np.array([]))
        elif len(flows) < self.w_len:
            return
        output = np.array([fs(f_0)] * len(flows))
        flows = np.array(flows).reshape(-1, self.w_len)
        flow_means = np.mean(flows, axis=1)
        flow_mean_deltas = flow_means[1:] - flow_means[:-1]
        flow_stds = np.std(flows, axis=1)
        if reporter != None:
            register_reporter(reporter)
        for i in atpbar(range(len(flow_means)-1), name="Labelling flow states"):
            if flow_stds[i] < self.f_thresh:
                w_std_i = 0.5 * self.f_thresh
            else:
                w_std_i = flow_stds[i]
            w_mean_delta = flow_mean_deltas[i]
            w_mean_i_1 = flow_means[i+1]
            curpos = (i+1) * self.w_len
            if abs(w_mean_delta) < 2 * w_std_i:
                if w_mean_i_1 > self.f_base + 2 * w_std_i:
                    # Process Peak Inspiratory Flow
                    output[curpos:curpos + self.w_len] = fs.peak_inspiratory_flow
                elif w_mean_i_1 < self.f_base - 2 * w_std_i:
                    # Process Peak Expiratory Flow
                    output[curpos:curpos + self.w_len] = fs.peak_expiratory_flow
                else:
                    # Process No Flow
                    output[curpos:curpos + self.w_len] = fs.no_flow
            elif w_mean_i_1 > self.f_base:
                if w_mean_delta > 0:
                    # Process Inspiration Initiation
                    output[curpos:curpos + self.w_len] = fs.inspiration_initiation
                else:
                    # Process Inspiration Termination
                    output[curpos:curpos + self.w_len] = fs.inspiration_termination
            else:
                if w_mean_delta < 0:
                    # Process Expiration Initiation
                    output[curpos:curpos + self.w_len] = fs.expiration_initiation
                else:
                    # Process Expiration Termination
                    output[curpos:curpos + self.w_len] = fs.expiration_termination
        if con is not None:
            con.put(output)
        else:
            self.f_labels = np.concatenate([self.f_labels, output])
        
    def process(self, pressures, flows, p_0=ps.peep, f_0=fs.no_flow):
        """
        Maps data points from pressure and flow to enumerated states
        
        Parameters
        ----------
        pressures : array like of real
            Pressure data points
        flows : array like of real
            Flow data points
        p_0 : PressureStates enum, optional
            The initial pressure state the program assumes it is in. Defaults to peep
        f_0 : FlowStates enum, optional
            The initial flow state the program assumes it is in. Defaults to no flow.
        
        Returns
        -------
        (array like of PressureStates enum, Array like of FlowStates enum)
        """
        buffer = len(pressures) % self.w_len
        if cpu_count() > 2:
            reporter = find_reporter()
            p_queue = Queue()
            f_queue = Queue()
            if buffer != 0:
                p_process = Process(target = self.process_pressures, args=(pressures[:-buffer], p_0, p_queue, reporter))
                f_process = Process(target = self.process_flows, args=(flows[:-buffer], f_0, f_queue, reporter))
            else:
                p_process = Process(target = self.process_pressures, args=(pressures, p_0, p_queue, reporter))
                f_process = Process(target = self.process_flows, args=(flows, f_0, f_queue, reporter))
            p_process.start()
            f_process.start()
            self.p_labels = np.concatenate((self.p_labels, p_queue.get()))
            self.f_labels = np.concatenate((self.f_labels, f_queue.get()))
            p_process.join()
            f_process.join()
            self.p_labels = np.concatenate([self.p_labels, np.array([self.p_labels[-1]] * buffer)])
            self.f_labels = np.concatenate([self.f_labels, np.array([self.f_labels[-1]] * buffer)])
        else:
            if buffer != 0:
                self.process_pressures(pressures[:-buffer], p_0)
                self.process_flows(flows[:-buffer], f_0)
            else:
                self.process_pressures(pressures, p_0)
                self.process_flows(flows, f_0)
            self.p_labels = np.concatenate([self.p_labels, np.array([self.p_labels[-1]] * buffer)])
            self.f_labels = np.concatenate([self.f_labels, np.array([self.f_labels[-1]] * buffer)])
