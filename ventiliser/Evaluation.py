#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:48:30 2020

@author: David Chong Tian Wei
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from ventiliser.BreathVariables import BreathVariables

class Evaluation:
    """
    Class to help visualise and evaluate breaths extracted from a record.
    
    Attributes
    ----------
    pressures : Array like of real
        Pressure data points for a record
    flows : Array like of real
        Flow data points for a record
    breaths : Array like of BreathVariables
        Breaths as calculating using mapper and phaselabeller
    freq : real
        Frequency in Hz of the recording sampling rate
        
    """
    @staticmethod
    def load_breaths_from_csv(path):
        """
        Utility method to load a csv file output from PhaseLabeller.get_breaths_raw to a list of BreathVariables
        
        Parameters
        ----------
        path : string
            Path to the raw breaths file
        
        Returns
        -------
        array like of BreathVariables objects
        """
        csv = pd.read_csv(path)
        breaths = csv.apply(Evaluation.__breathvariables_from_row, axis=1)
        return breaths
    @staticmethod
    def __breathvariables_from_row(x):
        """
        Helper method to apply on pandas dataframe
        
        Parameters
        ----------
        x : Pandas Series
            A row from a pandas dataframe containing breaths
        
        Returns
        -------
        BreathVariables object
        """
        output = BreathVariables()
        for attr in x.index:
            setattr(output, attr, int(x[attr]))
        return output
    
    def __init__(self, pressures, flows, breaths, freq):
        """
        Initialises the evaluation object with the data, predicted breaths, and frequency of the record
        
        Parameters
        ----------
        pressures : Array like of real
            Pressure data points for a record
        flows : Array like of real
            Flow data points for a record
        breaths : Array like of BreathVariables
            Breaths as calculating using mapper and phaselabeller
        freq : real
            Frequency in Hz of the recording sampling rate
        
        Returns
        -------
        None
        """
        
        self.pressures = np.array(pressures)
        self.flows = np.array(flows)
        self.breaths = breaths
        self.freq = freq
    
    def compare(self, labels, breath_attr):
        """
        Compares an attribute from the currently loaded breaths with a list of values. Identifies the label which is the closest match to each breath.
        
        Parameters
        ----------
        labels : array like of int
            A list of indices that you wish to compare with the currently loaded breaths
        breath_attr : string
            A BreathVariables attribute you wish to perform the comparison on
        
        Returns
        -------
        Pandas Dataframe
            A dataframe containing the closest matching breath to the given labels based on the attribute along with the difference
        """
        if self.breaths is None:
            print("No breaths to compare to")
            return
        labels = np.array(labels)
        output = []
        for breath in self.breaths:
            delta = abs(labels - getattr(breath, breath_attr))
            best = np.argmin(np.array(delta))
            output += [{"breath_index" : self.breaths.index(breath), "label_index" : best, "delta" : delta[best]}]
        return pd.DataFrame(output)
    
    def get_comparison_stats(self, labels, breath_attr):
        """
        Compares labels against an attribute of the loaded breaths. If multiple breaths map to the same label, the one with the smallest delta is kept and the remaining considered false positives.
        
        Parameters
        ----------
        labels : array like of int
            A list of indices that you wish to compare with the currently loaded breaths
        breath_attr : string
            A BreathVariables attribute you wish to perform the comparison on
        
        Returns
        -------
        dict
            A dictionary containing total number of breaths, number of matched breaths, total number of labels, number of matched labels, mean delta of all matches, std delta of all matches
        """
        comparison = self.compare(labels, breath_attr)
        mapped_labels = comparison["label_index"].unique()
        matches = []
        for label in mapped_labels:
            mapped_breaths = comparison.loc[comparison["label_index"] == label].sort_values(by=["delta"], ascending=True)
            matches += [mapped_breaths["delta"].iat[0]]
        return {"n_predicted" : len(self.breaths),
                  "n_predicted_matched" : len(matches),
                  "n_labels" : len(labels),
                  "n_labels_matched" : len(mapped_labels),
                  "delta_mean" : np.mean(matches),
                  "delta_std" : np.std(matches)}
        
    def plot_breath(self, index, attrs=[], p_y="Pressure", f_y="Flow", v_y="Volume", show=True):
        """
        Plots the pressure, volume and flow of a breath and allows labelling of key points
        
        Parameters
        ----------
        index : int
            Index of the breath you wish to plot
        attrs : array like of string, optional
            List of attributes you wish to annotate onto the breath (See BreathVariables for available attributes). Defaults to empty list
        p_y : string, optional
            Label for pressure plot y axis. Defaults to 'Pressure'.
        f_y : string, optional
            Label for flow plot y axis. Defaults to 'Flow'
        v_y : string, optional
            Label for volume plot y axis. Defaults to 'Volume'
        show : boolean, optional
            Flag to decide whether to show the plot or leave it so the user can add more to it. Defaults to True
        
        Returns
        -------
        None
        """
        if self.breaths is None:
            print("No breaths loaded")
        if index >= len(self.breaths):
            print("Index is larger than the number of breaths available")
        
        volume = [0]
        for f in self.flows[self.breaths[index].breath_start:self.breaths[index].breath_end]:
            volume.append(volume[-1] + f)
        
        x_values = np.array(range(0,self.breaths[index].breath_end - self.breaths[index].breath_start)) / self.freq
        plt.figure(figsize=(10,6))
        # Plot pressures
        plt.subplot(3,1,1)
        plt.title("Breath " + str(index))
        plt.plot(x_values,self.pressures[self.breaths[index].breath_start:self.breaths[index].breath_end], color="r")
        plt.ylim(bottom=0)
        plt.ylabel(p_y)
        for attr in attrs:
            xpos = x_values[getattr(self.breaths[index], attr) - self.breaths[index].breath_start]
            plt.axvline(x=xpos, color="k", linestyle="--")
            plt.text(xpos, random.uniform(plt.ylim()[0] * 0.9, plt.ylim()[1] * 0.9), attr)
        # Plot flows
        plt.subplot(3,1,2)
        plt.plot(x_values, self.flows[self.breaths[index].breath_start:self.breaths[index].breath_end], color="g")
        plt.axhline(y=0, color="k", linestyle="--")
        plt.ylabel(f_y)
        for attr in attrs:
            xpos = x_values[getattr(self.breaths[index], attr) - self.breaths[index].breath_start]
            plt.axvline(x=xpos, color="k", linestyle="--")
            plt.text(xpos, random.uniform(plt.ylim()[0] * 0.9, plt.ylim()[1] * 0.9), attr)
        # Plot volume
        plt.subplot(3,1,3)
        plt.plot(x_values, volume[1:], color="b")
        plt.xlabel("Time [s]")
        plt.ylabel(v_y)
        for attr in attrs:
            xpos = x_values[getattr(self.breaths[index], attr) - self.breaths[index].breath_start]
            plt.axvline(x=xpos, color="k", linestyle="--")
            plt.text(xpos, random.uniform(plt.ylim()[0] * 0.9, plt.ylim()[1] * 0.9), attr)
        
        plt.tight_layout()
        if show:
            plt.show()
    
    def plot_pv_loop(self, index, p_label="Pressure", v_label="Volume", show=True):
        """
        Plots the pressure-volume loop of the breath at index
        
        Parameters
        ----------
        index : int
            The index of the breath you wish to plot
        p_label : string, optional
            The label for the pressure axis. Defaults to 'Pressure'
        v_label : string, optional
            The label for the volume axis. Defaults to 'Volume'
        show : boolean, optional
            Flag for deciding whether to show the plot or leave it so the user can add more to it. Defaults to True
        
        Returns
        -------
        None
        """
        if self.breaths is None:
            print("No breaths loaded")
        if index >= len(self.breaths):
            print("Index is larger than the number of breaths available")
        volume = [0]
        for f in self.flows[self.breaths[index].breath_start:self.breaths[index].breath_end]:
            volume.append(volume[-1] + f)
        plt.plot(self.pressures[self.breaths[index].breath_start:self.breaths[index].breath_end], volume[1:])
        plt.xlabel(p_label)
        plt.ylabel(v_label)
        plt.title("Breath " + str(index))
        if show:
            plt.show()
            
    def plot_breaths_flow_phases(self, 
                                 attrs=["inspiration_initiation_length", "peak_inspiratory_flow_length", "inspiration_termination_length",
                                        "inspiratory_hold_length", "expiration_initiation_length", "peak_expiratory_flow_length",
                                        "expiration_termination_length","expiratory_hold_length"]):
        """
        Plots a stacked barplot of each breath where the stack is the length of each phase of the breath
        
        Parameters
        ----------
        attrs : array like of string, optional
            List of breath phases you wish to plot out. Defaults to all breath phases (see BreathVariables for more information)
        
        Returns
        -------
        None
        """
        df = pd.DataFrame([vars(x) for x in self.breaths])
        order = np.argsort(np.array(df["breath_end"] - df["breath_start"]))
        df = df.iloc[order,:]
        bottom = np.array([0] * df.shape[0])
        ind = np.arange(df.shape[0])
        for attr in attrs:
            plt.bar(ind, df[attr], 1, bottom = bottom, label=attr)
            bottom += df[attr].values
        plt.legend()
        plt.xlabel("Breaths")
        plt.ylabel("Length")
        plt.show()
        
