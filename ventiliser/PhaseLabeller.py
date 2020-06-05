#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:41:13 2020

@author: David Chong Tian Wei
"""
import math
import numpy as np
import pandas as pd
from atpbar import atpbar
from ventiliser.FlowStates import FlowStates as fs
from ventiliser.PressureStates import PressureStates as ps
from ventiliser.BreathVariables import BreathVariables

class PhaseLabeller:
    """ Class for segmenting and labelling breath sub-phases
    
    Attributes
    ----------
    freq : real
        Sampling rate for the record to be processed
    hold_length : integer
        Threshold in number of data points for the amount of time a breath can be in a no flow state before considered termintated
    leak_perc_thresh : real
        The proportion of leak permitted before breath is conidered physiologically implausible and to be flagged for merging in post processing
    exp_hold_len : integer
        The time in number of data points of the expiratory hold that must occur between breaths to deflag for merging in post-processing
    permit_double_cycling : boolean
        Decide whether to merge double cycles in post-processing
    
    Methods
    -------
    configure(w_len=3, freq=100, hold_length=0.5, leak_perc_thresh=0.33, exp_hold_len=0.05, permit_double_cycling = False)
        Set constants for processing
    process(p_labels, f_labels, pressures, flows, post_processing=True)
        Performs the segmentation and phase labelling
    get_breaths(length_units)
        Returns the segmented breaths and calculated features as a pandas dataframe modifying length variables given the sampling rate. See BreathVariables for list of variables returned
    get_breaths_raw()
        Returns the segmented breaths and calculated features as a pandas dataframe. See BreathVariables for list of variables returned
    """
    def __init__(self):
        self.configure()
    
    def configure(self, w_len=3, freq=100, hold_length=0.5, leak_perc_thresh=0.66, exp_hold_len=0.05, permit_double_cycling = False):
        """ Sets the constants for segmentation and post-processing
        
        :param w_len: Size of window used when mapping states
        :type w_len: integer
        :param freq: Sampling rate for the record to be processed
        :type freq: real
        :param hold_length: Threshold in seconds for the amount of time a breath can be in a no flow state before considered termintated
        :type hold_length: real
        :param leak_perc_thresh: The proportion of leak permitted before breath is conidered physiologically implausible and to be flagged for merging in post processing
        :type leak_perc_thresh: real
        :param exp_hold_len: The time in seconds of the expiratory hold that must occur between breaths to deflag for merging in post-processing
        :type exp_hold_len: real
        :param permit_double_cycling: Decide whether to merge double cycles in post-processing
        :type permit_double_cycling: boolean
        """
        self.breaths = []
        self.w_len = w_len
        self.freq = freq
        self.max_hold = math.ceil(freq * hold_length)
        self.leak_perc_thresh = leak_perc_thresh
        self.exp_hold_len = math.ceil(freq * exp_hold_len)
        self.permit_double_cycling = permit_double_cycling
    
    def process(self, p_labels, f_labels, pressures, flows, post_processing=True):
        """Given the pressure and flow data points and labels, segments the data into breaths, identifies respiratory sub-phases, and calculates some physiological values
        
        :param p_labels: The PressureStates labels generated from StateMapper
        :type p_labels: Array like of PressureStates enum
        :param f_labels: The FlowStates labels generated from StateMapper
        :type f_labels: Array like of FlowStates enum
        :param pressures: Pressure data points
        :type pressures: Array like of real
        :param flows: Flow data points
        :type flows: Array like of real
        :returns: None
        :rtype: None
        """
        if type(pressures) is not np.array:
            pressures = np.array(pressures)
        if type(flows) is not np.array:
            flows = np.array(flows)
        print("Segmenting into breaths")
        self.breaths += [BreathVariables()]
        self.breaths[-1].breath_end = 0
        while(self.breaths[-1].breath_end != len(f_labels)):
            self.breaths += [self.__get_next_breath(f_labels, self.breaths[-1].breath_end)]
        # First and last breaths are usually inaccurate
        if len(self.breaths) > 1:
            self.breaths = self.breaths[1:]
            print(str(len(self.breaths)) + " breaths identified")
            for i in atpbar(range(len(self.breaths)), name="Processing breaths"):
                self.breaths[i].breath_number = i+1
                self.__information_approach(p_labels, f_labels, self.breaths[i])
                self.__calculate_features(self.breaths[i], pressures, flows)
            if post_processing:
                self.__post_process(p_labels, f_labels, pressures, flows)
        else:
            self.breaths = []
            print("Warning: No breaths identified")
    
    def get_breaths(self, length_units="ms"):
        """Returns the segmented breaths and calculated features as a pandas dataframe. See BreathVariables for list of variables returned
        
        :param length_units: Unit to use for length calculations, accepts 'ms' and 's' for milliseconds and seconds respectively
        :type length_units: string
        :returns: Table of segmented breaths and charactersitics for each breath
        :rtype: Pandas Dataframe
        """
        df = pd.DataFrame([vars(x) for x in self.breaths])
        if length_units == "ms":
            df[list(filter(lambda x : "length" in x, df.columns))] *= 1000 / self.freq
        elif length_units == "s":
            df[list(filter(lambda x : "length" in x, df.columns))] *= 1 / self.freq
        return df[["breath_number", "breath_start", "breath_end", "inspiration_initiation_start", "peak_inspiratory_flow_start",
                  "inspiration_termination_start", "inspiratory_hold_start", "expiration_initiation_start",	"peak_expiratory_flow_start",
                  "expiration_termination_start", "expiratory_hold_start", "pressure_rise_start", "pip_start", "pressure_drop_start",
                  "peep_start", "inspiration_initiation_length", "peak_inspiratory_flow_length",
                  "inspiration_termination_length", "inspiratory_hold_length", "expiration_initiation_length", "peak_expiratory_flow_length",
                  "expiration_termination_length", "expiratory_hold_length", "pressure_rise_length", "pip_length", "pressure_drop_length",
                  "peep_length", "pip_to_no_flow_length", "peep_to_no_flow_length", "lung_inflation_length", "total_inspiratory_length",
                  "lung_deflation_length", "total_expiratory_length", "inspiratory_volume", "expiratory_volume", "max_inspiratory_flow",
                  "max_expiratory_flow", "max_pressure", "min_pressure", "pressure_flow_correlation"]]
        
    def get_breaths_raw(self):
        """Returns the raw breath calculations without modifying for time interval
        
        :returns: Dataframe containing breaths and calculated variables
        :rtype: Pandas Dataframe
        """
        return pd.DataFrame([vars(x) for x in self.breaths])
    
    def get_breath_annotations(self, N, p_states=list(ps), f_states=list(fs)):
        """ Returns a Nx3 dataframe containing key points of breaths mapped to indices to be used with GUI annotator for viewing
            
        :param N: Length of the sample that was analyzed (in terms of data points)
        :type N: Integer
        :param p_states: The pressure states from each breath that you would like mapped
        :type p_states: array like of PressureStates
        :param f_states: The flow states from each breath that you woudld like mapped
        :type f_states: array like of FlowStates
        :returns: Dataframe containing keypoints at each index of the data on which the analysis was performed
        :rtype: Pandas Dataframe
        """
        output = pd.concat([pd.Series(range(N)), pd.Series([-1] * N), pd.Series([-1] * N)], axis=1)
        output.columns = ["index","pressure_annotations","flow_annotations"]
        breaths = self.get_breaths_raw()
        for p in p_states:
            if p == ps.pressure_rise:
                output.iloc[breaths["pressure_rise_start"]-1,1] = ps.pressure_rise.value
            elif p == ps.pip:
                output.iloc[breaths["pip_start"]-1,1] = ps.pip.value
            elif p == ps.pressure_drop:
                output.iloc[breaths["pressure_drop_start"]-1,1] = ps.pressure_drop.value
            elif p == ps.peep:
                output.iloc[breaths["peep_start"]-1,1] = ps.peep.value
        for f in f_states:
            if f == fs.inspiration_initiation:
                output.iloc[breaths["inspiration_initiation_start"]-1,2] = fs.inspiration_initiation.value
            elif f == fs.peak_inspiratory_flow:
                output.iloc[breaths["peak_inspiratory_flow_start"]-1,2] = fs.peak_inspiratory_flow.value
            elif f == fs.inspiration_termination:
                output.iloc[breaths["inspiration_termination_start"]-1,2] = fs.inspiration_termination.value
            elif f == fs.no_flow:
                output.iloc[breaths["inspiratory_hold_start"]-1,2] = fs.no_flow.value
                output.iloc[breaths["expiratory_hold_start"]-1,2] = fs.no_flow.value
            elif f == fs.expiration_initiation:
                output.iloc[breaths["expiration_initiation_start"]-1,2] = fs.expiration_initiation.value
            elif f == fs.peak_expiratory_flow:
                output.iloc[breaths["peak_expiratory_flow_start"]-1,2] = fs.peak_expiratory_flow.value
            elif f == fs.expiration_termination:
                output.iloc[breaths["expiration_termination_start"]-1,2] = fs.expiration_termination.value
        output = output.loc[output.iloc[:,1:].apply(lambda x : x.sum() != -2, axis=1)]
        return output
    
    def __get_next_breath(self, labels, start):
        """Identifies the next breath in the record based on Inspiration-Inspiration interval
        
        :param labels: The array of flow labels calculated from a StateMapper object
        :type labels: Array like of FlowStates enum
        :param start: Index from which to start searching for a breath
        :type start: integer
        :param max_hold: Maximum number of datapoints that no_flow is encountered before the breath is deemed to be an outlier so we cut it off at the next inspiration event.
        :type max_hold: integer
        :returns: A breath object containing the start and end points
        :rtype: BreathVariables object
        """
        running_hold = 0
        expiration_encountered = False
        for i in range(start, len(labels)):
            if labels[i] is fs.inspiration_initiation or labels[i] is fs.peak_inspiratory_flow:
                running_hold = 0
                # If this is the start of the recording, the start index may be inaccurate so we reset it here
                if start == 0:
                    start = i
                    expiration_encountered=False
                if expiration_encountered:
                    breath = BreathVariables()
                    breath.breath_start = start
                    breath.breath_end = i
                    return breath
            elif labels[i] is fs.expiration_initiation or labels[i] is fs.peak_expiratory_flow or labels[i] is fs.expiration_termination or running_hold > self.max_hold:
                expiration_encountered = True
            elif labels[i] is fs.no_flow:
                running_hold += 1
                
        # If code reaches this point then it is the last breath of the record
        breath = BreathVariables()
        breath.breath_start = start
        breath.breath_end = len(labels)
        return breath

    def __sum_information(self, groups, target_classes):
        """Calculates the total information across multiple groups based on a target classes vs everything else
        
        :param groups: A list of arrays containing classes
        :type groups: Array like of Array like
        :param target_classes: An array of classes to consider as a single category when calculating information
        :type target_classes: Array like
        """
        totallen = np.sum([len(x) for x in groups])
        inf = 0
        for x in groups:
            xlen = len(x)
            class_sum = 0
            for target_class in target_classes:
                class_sum += np.sum(x == target_class)
            p = class_sum / xlen
            inf += -p * np.log(p + 1E-7) * xlen / totallen
            p = (xlen - class_sum) / xlen
            inf += -p * np.log(p + 1E-7) * xlen / totallen
        return inf
    
    def __maximise_information_gain(self, labels, target_classes):
        """Finds the split on the given labels which maximises information gain
        
        :param labels: An array of labels (enumerated flow/pressure states)
        :type labels: Array like
        :param target_classes: An array of labels (enumerated flow/pressure states) to use to calculate information
        :type target_classes: Array like
        """
        some_exists = False
        for target_class in target_classes:
            if target_class in labels:
                some_exists = True
                break
        if not some_exists:
            return (0, None, labels)
        idx = -1
        infgain = -1
        totalinf = self.__sum_information([labels], target_classes)
        for i in range(self.w_len, len(labels), self.w_len):
            remaininginf = self.__sum_information([labels[:i], labels[i:]], target_classes)
            if (infgain == -1 or infgain < totalinf - remaininginf) and not np.isnan(totalinf - remaininginf):
                infgain = totalinf - remaininginf
                idx = i
        if idx < 0:
            return (0, None, labels)
        else:
            return (idx, labels[:idx], labels[idx:])
    
    def __maximise_information_gain_backup(self, labels, target_classes):
        some_exists = False
        for target_class in target_classes:
            if target_class in labels:
                some_exists = True
                break
        if not some_exists:
            return (0, np.array([]), labels)
        if len(labels) == 0:
            return (0, np.array([]), labels)
        elif len(labels) == 1:
            for target_class in target_classes:
                return(1, labels, np.array([]))
            return (0, np.array([]), labels)
        else:
            pivot = len(labels) // 2
            left_inf = self.__sum_information([labels[:pivot]], target_classes)
            right_inf = self.__sum_information([labels[pivot:]], target_classes)
            if left_inf > right_inf:
                idx, prev_labels, post_labels = self.__maximise_information_gain(labels[:pivot], target_classes)
                return (idx, prev_labels, np.concatenate([post_labels,labels[pivot:]]))
            else:
                idx, prev_labels, post_labels = self.__maximise_information_gain(labels[pivot:], target_classes)
                return(idx + pivot, np.concatenate([labels[:pivot], prev_labels]), post_labels)
       
    def __information_approach(self, p_labels, f_labels, breath):
        """Tries to identify sub-phases of each breath based on maximising information gain on splitting
        
        :param p_labels: Pressure labels for the record calculated using StateMapper
        :type p_labels: Array like of PressureStates enum
        :param f_labels: Flow labels for the record calculated using StateMapper
        :type f_labels: Array like of FlowStates enum
        :param breath: BreathVariables object for the breath to calculate sub phases
        :type breath: BreathVariables object
        :returns: None
        :rtype: None
        """
        p_labels = p_labels[breath.breath_start:breath.breath_end]
        f_labels = f_labels[breath.breath_start:breath.breath_end]
        labels = f_labels
        # Inspiration initiation at breath start by segmentation definition
        breath.inspiration_initiation_start = breath.breath_start
        # Find Peak inspiratory flow start by finding end of split
        breath.peak_inspiratory_flow_start, _, labels = self.__maximise_information_gain(labels, [fs.inspiration_initiation])
        breath.peak_inspiratory_flow_start += breath.breath_start
        # Find Inspiration termination start by finding end of split
        breath.inspiration_termination_start, _, labels = self.__maximise_information_gain(labels, [fs.peak_inspiratory_flow])
        breath.inspiration_termination_start += breath.peak_inspiratory_flow_start
        # Find Inspiratory Hold start by finding end of split
        breath.inspiratory_hold_start, _, labels = self.__maximise_information_gain(labels, [fs.inspiration_termination])
        breath.inspiratory_hold_start += breath.inspiration_termination_start
        # Find Peak expiratory flow start by finding end of split
        breath.peak_expiratory_flow_start, _, labels = self.__maximise_information_gain(labels, [fs.expiration_initiation])
        breath.peak_expiratory_flow_start += breath.inspiratory_hold_start
        # Find Expiration Termination Start by finding end of split
        breath.expiration_termination_start, _, labels = self.__maximise_information_gain(labels, [fs.peak_expiratory_flow])
        breath.expiration_termination_start += breath.peak_expiratory_flow_start
        # Find Expiratory hold start by finding start of split
        no_flow, _, labels = self.__maximise_information_gain(labels, [fs.no_flow])
        if no_flow == 0:
            breath.expiratory_hold_start = breath.breath_end
        else:
            breath.expiratory_hold_start = breath.expiration_termination_start + no_flow
        # Find expiration initiation start by finding end of split
        templabels = f_labels[breath.inspiratory_hold_start - breath.breath_start : breath.peak_expiratory_flow_start - breath.breath_start]
        breath.expiration_initiation_start, _, labels = self.__maximise_information_gain(templabels, [fs.no_flow])
        breath.expiration_initiation_start += breath.inspiratory_hold_start
        
        labels = p_labels
        # Find pip start by finding end of split
        breath.pip_start, _, labels = self.__maximise_information_gain(labels, [ps.pressure_rise])
        breath.pip_start += breath.breath_start
        # Find pressure drop start by finding end of split
        breath.pressure_drop_start, _, labels = self.__maximise_information_gain(labels, [ps.pip])
        breath.pressure_drop_start += breath.pip_start
        # Find peep start by finding start of split
        breath.peep_start, _, labels = self.__maximise_information_gain(labels, [ps.peep])
        breath.peep_start += breath.pressure_drop_start
        # Find pressure rise start by finding start of split
        breath.pressure_rise_start, _, labels = self.__maximise_information_gain(p_labels[:breath.pip_start - breath.breath_start], [ps.peep])
        breath.pressure_rise_start += breath.breath_start
    
    def __calculate_features(self, breath, pressures, flows):
        """Calculates the values relevant for physiology like tidal volumes and respiratory phase lengths
        
        :param breath: The breath for which to calculate the physiological values
        :type breath: BreathVariables
        :param pressures: Pressure data points for the record
        :type pressures: Array like of real
        :param flows: Flow data points for the record
        :type flows: Array like of real
        """
        p = np.array(pressures[breath.breath_start:breath.breath_end])
        f = np.array(flows[breath.breath_start:breath.breath_end])

        # Pressure phases
        breath.pressure_rise_length = breath.pip_start - breath.pressure_rise_start
        breath.pip_length = breath.pressure_drop_start - breath.pip_start
        breath.pressure_drop_length = breath.peep_start - breath.pressure_drop_start
        breath.peep_length = breath.breath_end - breath.peep_start
        
        # Flow phases
        breath.inspiration_initiation_length = breath.peak_inspiratory_flow_start - breath.inspiration_initiation_start
        breath.peak_inspiratory_flow_length = breath.inspiration_termination_start - breath.peak_inspiratory_flow_start
        breath.inspiration_termination_length = breath.inspiratory_hold_start - breath.inspiration_termination_start
        breath.inspiratory_hold_length = breath.expiration_initiation_start - breath.inspiratory_hold_start
        breath.expiration_initiation_length = breath.peak_expiratory_flow_start - breath.expiration_initiation_start
        breath.peak_expiratory_flow_length = breath.expiration_termination_start - breath.peak_expiratory_flow_start
        breath.expiration_termination_length = breath.expiratory_hold_start - breath.expiration_termination_start
        breath.expiratory_hold_length = breath.breath_end - breath.expiratory_hold_start
        breath.lung_inflation_length = breath.inspiratory_hold_start - breath.inspiration_initiation_start
        breath.total_inspiratory_length = breath.expiration_initiation_start - breath.inspiration_initiation_start
        breath.lung_deflation_length = breath.expiratory_hold_start - breath.expiration_initiation_start
        breath.total_expiratory_length = breath.breath_end - breath.expiration_initiation_start
        breath.pip_to_no_flow_length = breath.inspiratory_hold_start - breath.pip_start
        breath.peep_to_no_flow_length = breath.expiratory_hold_start - breath.peep_start
        
        # Volumes
        breath.inspiratory_volume = f[f > 0].sum()
        breath.expiratory_volume = np.abs(f[f < 0].sum())
        breath.max_inspiratory_flow = f.max()
        breath.max_expiratory_flow = f.min()
        
        # Pressures
        breath.max_pressure = p.max()
        breath.min_pressure = p.min()
        
        # Correlation
        breath.pressure_flow_correlation = np.corrcoef(p,f)[0,1]
    
    def __post_process(self, p_labels, f_labels, pressures, flows):
        """ Performs merging of adjacent breaths dependent on whether inspiration and expiration volumes match
        
        :param p_labels: Pressure labels for record
        :type p_labels: Array like of PressureStates enum
        :param f_labels: Flow labels for record
        :type f_labels: Array like of FlowStates enum
        :param pressures: Pressure data points
        :type pressures: Array like of real
        :param flows: Flow data points
        :type flows: Array like of real
        :returns: None
        :rtype: None
        """
        merged_breaths = [self.breaths[0]]
        begin_merge = False
        insp_sum = 0
        exp_sum = 0
        error_start = 0
        for i in atpbar(range(1,len(self.breaths)), name="Post-processing"):
            if not begin_merge:
                breath_leak_perc = (self.breaths[i].inspiratory_volume - self.breaths[i].expiratory_volume) / self.breaths[i].inspiratory_volume
                if abs(breath_leak_perc) > self.leak_perc_thresh:
                    if breath_leak_perc < 0 and self.breaths[i-1].expiratory_hold_length <= self.exp_hold_len:
                        error_start = i - 1
                        merged_breaths.pop()
                        begin_merge = True
                        insp_sum += self.breaths[i-1].inspiratory_volume + self.breaths[i].inspiratory_volume
                        exp_sum += self.breaths[i-1].expiratory_volume + self.breaths[i].expiratory_volume
                    elif breath_leak_perc > 0 and self.breaths[i-1].expiratory_hold_length <= self.exp_hold_len:
                        begin_merge = True
                        error_start = i
                        insp_sum += self.breaths[i].inspiratory_volume
                        exp_sum += self.breaths[i].expiratory_volume
                    else:
                        merged_breaths += [self.breaths[i]]
                else:
                    merged_breaths += [self.breaths[i]]
            else:
                if ((abs(insp_sum - exp_sum)/insp_sum < self.leak_perc_thresh or self.breaths[i-1].expiratory_hold_length > self.exp_hold_len) and error_start != i-1) or (self.breaths[i].pressure_flow_correlation > 0.2 and not self.permit_double_cycling):
                    # Begin to merge breaths
                    begin_merge = False
                    insp_sum = 0
                    exp_sum = 0
                    merged_breath = BreathVariables()
                    merged_breath.breath_start = self.breaths[error_start].breath_start
                    merged_breath.breath_end = self.breaths[i-1].breath_end
                    self.__information_approach(p_labels, f_labels, merged_breath)
                    self.__calculate_features(merged_breath, pressures, flows)
                    merged_breaths += [merged_breath]
                    # Check if current breath needs to be merged
                    breath_leak_perc = (self.breaths[i].inspiratory_volume - self.breaths[i].expiratory_volume) / self.breaths[i].inspiratory_volume
                    if abs(breath_leak_perc) > self.leak_perc_thresh:
                        if breath_leak_perc < 0 and self.breaths[i].expiratory_hold_length <= self.exp_hold_len:
                            merged_breaths.pop()
                            begin_merge = True
                            insp_sum += self.breaths[i-1].inspiratory_volume + self.breaths[i].inspiratory_volume
                            exp_sum += self.breaths[i-1].expiratory_volume + self.breaths[i].expiratory_volume
                        elif breath_leak_perc > 0 and self.breaths[i].expiratory_hold_length <= self.exp_hold_len:
                            begin_merge = True
                            error_start = i
                            insp_sum += self.breaths[i].inspiratory_volume
                            exp_sum += self.breaths[i].expiratory_volume
                        else:
                            merged_breaths += [self.breaths[i]]
                    else:
                        merged_breaths += [self.breaths[i]]
                else:
                    insp_sum += self.breaths[i].inspiratory_volume
                    exp_sum += self.breaths[i].expiratory_volume
        
        self.breaths = merged_breaths
        for i in atpbar(range(len(self.breaths)), name="Re-numbering breaths"):
            self.breaths[i].breath_number = i+1
