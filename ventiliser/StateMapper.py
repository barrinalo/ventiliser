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
        
    Methods
    -------
    configure(p_base=5.5, f_base=0, f_thresh=0.1, freq=100, t_len=0.03)
        Configures attributes required for processing a record
    process_pressures(pressures, p_0=PressureStates.peep, display_progress=False, con=None, reporter=None)
        Calcultes the pressure state labels for the data given in pressures.
    process_flows(flows, f_0=FlowStates.no_flow, display_progress=False, con=None, reporter=None)
        Calculates the flow state labels for the data given in flows.
    process(pressures, flows, p_0=PressureStates.peep, f_0=FlowStates.no_flow, display_progress=False, single_core=False)
        Calculates the pressure and flow state labels for data given in pressures and flows.
    """
    
    def __init__(self):
        self.p_labels = np.array([])
        self.f_labels = np.array([])
        self.configure();
    
    def configure(self, p_base=5.5, f_base=0, f_thresh=0.1, freq=100, t_len=0.03):
        """ Sets processing constants. To be called before process
        
        :param p_base: Baseline pressure / PEEP set on the ventilator
        :type p_base: real
        :param f_base: Baseline flow which is usually 0
        :type f_base: real
        :param f_thresh: Threshold for the standard deviation of a window to be considered non-stationary
        :type f_thresh: real
        :param freq: Sampling rate of the record to be processed
        :type freq: real
        :param t_len: Length of time in seconds of the desired window to use
        :type t_len: real
        :returns: None
        :rtype: None
        """
        self.p_base = p_base
        self.f_base = f_base
        self.f_thresh = f_thresh
        self.freq = freq
        self.w_len = math.ceil(freq * t_len)
        if self.w_len < 3:
            print("Warning: calculated window length is less than 3, average and standard deviation calculations may be unhelpful, consider increasing t_len")
    
    def get_labels(self):
        """Returns the calculated labels
        
        :returns: Dataframe containing the calculated pressure and flow states in value form
        :rtype: Pandas dataframe of integer
        """
        return pd.DataFrame({"Pressure_States" : [x.value for x in self.p_labels], "Flow_States" : [x.value for x in self.f_labels]})
    
    def process_pressures(self, pressures, p_0=ps.peep, con=None, reporter=None):
        """Maps data points from pressure to enumerated states
        
        :param pressures: Pressure data points
        :type pressures: Array like of real
        :param p_0: Enumerated pressure state for padding
        :type p_0: PressureStates enum
        :param display_progress: Determines if progress should be shown
        :type display_progress: boolean
        :param con: Queue data structure for multiprocessing
        :type con: Queue
        :param reporter: For displaying progress bar in multiprocessing
        :type reporter: reporter
        :returns: None
        :rtype: None
        """
        if len(pressures) < self.w_len and con is None:
            con.put(np.array([]))
        elif len(pressures) < self.w_len:
            return
        
        if type(pressures) is not np.array:
            p = np.array(pressures)
        else:
            p = pressures
        # Pad before begin processing
        p_labels = [p_0] * self.w_len
        w_mean_i = np.mean(p[:self.w_len])
        w_std_i = np.std(p[:self.w_len])
        p_hold = self.p_base
        if reporter != None:
            register_reporter(reporter)
        for i in atpbar(range(self.w_len, len(p), self.w_len), name="Labelling pressure states"):
            w_mean_i, w_std_i, p_label, p_hold = self.__calculate_pressure_window(w_mean_i, w_std_i, p_labels[-1], p_hold, p, i)
            p_labels += [p_label] * self.w_len
        
        if len(self.p_labels) > 0:
            p_labels = p_labels[self.w_len-1:]
            
        if con is not None:
            con.put(p_labels)
        else:
            self.p_labels = np.concatenate((self.p_labels, p_labels))
            
    def __calculate_pressure_window(self, w_mean_i, w_std_i, prev_label, prev_pressure_hold, p, i_1):
        """Calculates pressure mean and std for current window and compares it with previous window
        
        :param w_mean_i: Mean of the previous window pressure
        :type w_mean_i: real
        :param w_std_i: Standard deviation of the previous window pressure
        :type w_std_i: real
        :param prev_label: Previous pressure label
        :type prev_label: PressureStates enum
        :param prev_pressure_hold: Previous pressure when a hold state occurred
        :type prev_pressure_hold: real
        :param p: Pressure data points
        :type p: Array like of real
        :param i_1: The index for the current window
        :type i_1: integer
        :returns: 4-Tuple containing current window mean, standard deviation, pressure state enum, and previous pressure hold
        :rtype: (real, real, PressureStates enum, real)
        """
        w_mean_i_1 = np.mean(p[i_1:i_1 + self.w_len])
        w_std_i_1 = np.std(p[i_1:i_1 + self.w_len])
        w_mean_delta = w_mean_i_1 - w_mean_i
        
        # If standard deviation is too small, set it to a minimum threshold
        if w_std_i < 0.1 * self.p_base:
            w_std_i = self.p_base * 0.05
        
        # Process stationary states
        if abs(w_mean_delta) < 2 * w_std_i:
            # Process for PIP
            if w_mean_i_1 > (self.p_base + prev_pressure_hold) / 2 + 2 * w_std_i:
                if prev_label is not ps.pip:
                    return (w_mean_i_1, w_std_i_1, ps.pip, w_mean_i_1)
                else:
                    return (w_mean_i_1, w_std_i_1, ps.pip, prev_pressure_hold)
            else:
                # Process PEEP
                if prev_label is not ps.peep:
                    return (w_mean_i_1, w_std_i_1, ps.peep, w_mean_i_1)
                else:
                    return (w_mean_i_1, w_std_i_1, ps.peep, prev_pressure_hold)
        elif w_mean_delta > 0:
            # Process pressure rise
            return (w_mean_i_1, w_std_i_1, ps.pressure_rise, prev_pressure_hold)
        else:
            # Process pressure drop
            return (w_mean_i_1, w_std_i_1, ps.pressure_drop, prev_pressure_hold)
        
    def process_flows(self, flows, f_0=0, con=None, reporter=None):
        """Maps data points from pressure to enumerated states
        
        :param flows: Flow data points
        :type flows: Array like of real
        :param f_0: Enumerated flow state for padding
        :type f_0: integer
        :param display_progress: Determines if progress should be shown
        :type display_progress: boolean
        :param con: Queue data structure for multiprocessing
        :type con: Queue
        :param reporter: For displaying progress bar in multiprocessing
        :type reporter: reporter
        :returns: None
        :rtype: None
        """
        if len(flows) < self.w_len and con is None:
            con.put(np.array([]))
        elif len(flows) < self.w_len:
            return
        
        if type(flows) is not np.array:
            f = np.array(flows)
        else:
            f = flows
        
        # Pad before begin processing
        f_labels = [f_0] * self.w_len
        w_mean_i = np.mean(f[:self.w_len])
        w_std_i = np.std(f[:self.w_len])
        if reporter != None:
            register_reporter(reporter)
        for i in atpbar(range(self.w_len, len(f), self.w_len), name="Labelling flow states"):
            w_mean_i, w_std_i, f_label = self.__calculate_flow_window(w_mean_i, w_std_i, f_labels[-1], f, i)
            f_labels += [f_label] * self.w_len
            
        if len(self.f_labels) > 0:
            f_labels = f_labels[self.w_len-1:]
            
        if con is not None:
            con.put(f_labels)
        else:
            self.f_labels = np.concatenate((self.f_labels, f_labels))
    
    def __calculate_flow_window(self, w_mean_i, w_std_i, prev_label, f, i_1):
        """Calculates flow mean and std for current window and compares it with previous window
        
        :param w_mean_i: Mean of the previous window flow
        :type w_mean_i: real
        :param w_std_i: Standard deviation of the previous window flow
        :type w_std_i: real
        :param prev_label: Previous flow label
        :type prev_label: FlowStates enum
        :param p: Flow data points
        :type p: Array like of real
        :param i_1: The index for the current window
        :type i_1: integer
        :returns: 3-Tuple containing current window mean, standard deviation, and flow state enum
        :rtype: (real, real, FlowStates enum)
        """
        w_mean_i_1 = np.mean(f[i_1:i_1 + self.w_len])
        w_std_i_1 = np.std(f[i_1:i_1 + self.w_len])
        w_mean_delta = w_mean_i_1 - w_mean_i
        if w_std_i < self.f_thresh:
            w_std_i = self.f_thresh
        
        # Process stationary states
        if abs(w_mean_delta) < 2 * w_std_i:
            if w_mean_i_1 > self.f_base + 2 * w_std_i:
                # Process Peak Inspiratory Flow
                return (w_mean_i_1, w_std_i_1, fs.peak_inspiratory_flow)
            elif w_mean_i_1 < self.f_base - 2 * w_std_i:
                # Process Peak Expiratory Flow
                return (w_mean_i_1, w_std_i_1, fs.peak_expiratory_flow)
            else:
                # Process No Flow
                return (w_mean_i_1, w_std_i_1, fs.no_flow)
        elif w_mean_i_1 > self.f_base:
            if w_mean_delta > 0:
                # Process Inspiration Initiation
                return(w_mean_i_1, w_std_i_1, fs.inspiration_initiation)
            else:
                # Process Inspiration Termination
                return(w_mean_i_1, w_std_i_1, fs.inspiration_termination)
        else:
            if w_mean_delta < 0:
                # Process Expiration Initiation
                return(w_mean_i_1, w_std_i_1, fs.expiration_initiation)
            else:
                # Process Inspiration Termination
                return(w_mean_i_1, w_std_i_1, fs.expiration_termination)
        
    def process(self, pressures, flows, p_0=ps.peep, f_0=fs.no_flow):
        """Maps data points from pressure and flow to enumerated states
        
        :param pressures: Pressure data points
        :type pressures: Array like of real
        :param flows: Flow data points
        :type flows: Array like of real
        :param p_0: Enumerated pressure state for padding
        :type p_0: PressureStates enum
        :param f_0: Enumerated flow state for padding
        :type f_0: FlowStates enum
        :returns: 2-Tuple containing pressure and flow states
        :rtype: (Array like PressureStates enum, Array like FlowStates enum)
        """
        if cpu_count() > 2:
            reporter = find_reporter()
            p_queue = Queue()
            f_queue = Queue()
            p_process = Process(target = self.process_pressures, args=(pressures, p_0, p_queue, reporter))
            f_process = Process(target = self.process_flows, args=(flows, f_0, f_queue, reporter))
            p_process.start()
            f_process.start()
            self.p_labels = np.concatenate((self.p_labels, p_queue.get()))
            self.f_labels = np.concatenate((self.f_labels, f_queue.get()))
            p_process.join()
            f_process.join()
        else:
            self.process_pressures(pressures, p_0)
            self.process_flows(flows, f_0)
        if len(self.p_labels) > len(pressures):
            self.p_labels = self.p_labels[:len(pressures)]
        if len(self.f_labels) > len(flows):
            self.f_labels = self.f_labels[:len(flows)]
