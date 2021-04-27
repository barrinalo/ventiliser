#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 12:31:52 2020

@author: David Chong Tian Wei
"""
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QDialog, QDialogButtonBox, QAction, QTableView, QFileDialog, QLabel, QLineEdit, QHBoxLayout, QComboBox, QWidget, QPushButton
from PyQt5.QtCore import Qt, QAbstractTableModel
import pyqtgraph as pg
import os
import pandas as pd
import numpy as np
from ventiliser.FlowStates import FlowStates as fs
from ventiliser.PressureStates import PressureStates as ps

class ChangePointAnnotator:
    def __init__(self):
        self.app = QApplication([])
        self.root = ChangePointAnnotatorWindow()
        self.root.show()
        self.app.exec()

class BreathAnnotator:
    def __init__(self):
        self.app = QApplication([])
        self.root = BreathAnnotatorWindow()
        self.root.show()
        self.app.exec()

class BreathAnnotatorWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Breath Annotation Tool")
        self.resize(1280,720)
        pg.setConfigOption('foreground', "k")
        pg.setConfigOption('background', 'w')
        
        self.current_breath = 0
        
        # Label choices
        self.info_layout = QVBoxLayout()
        self.info_layout.setAlignment(Qt.AlignTop)
        self.cur_breath_display = QLabel("Current breath is {}".format(self.current_breath))
        self.cur_breath_label = QLabel("Current breath label is ")
        self.next_breath = QPushButton("Next Breath")
        self.next_breath.clicked.connect(self.__next_breath)
        self.prev_breath = QPushButton("Previous Breath")
        self.prev_breath.clicked.connect(self.__prev_breath)
        self.info_layout.addWidget(self.cur_breath_display)
        self.info_layout.addWidget(self.cur_breath_label)
        self.info_layout.addWidget(self.next_breath)
        self.info_layout.addWidget(self.prev_breath)
        self.info_layout.addWidget(QLabel("Breath Label Options"))
        
        # Graphing
        self.display = pg.GraphicsView()
        self.display_layout = pg.GraphicsLayout()
        self.display.setCentralItem(self.display_layout)
        self.pressure_plot = self.display_layout.addPlot(title="Pressure waveform")
        self.pressure_plot.showGrid(x=True, y=True)
        self.pressure_plot.setMenuEnabled(False)
        self.pressure_plot.setLabel("left", "Pressure")
        self.pressure_plot.setLabel("bottom", "Time")
        self.pressure_plot.setClipToView(True)
        self.display_layout.nextRow()
        self.flow_plot = self.display_layout.addPlot(title="Flow waveform")
        self.flow_plot.showGrid(x=True, y=True)
        self.flow_plot.setMenuEnabled(False)
        self.flow_plot.setLabel("left", "Flow")
        self.flow_plot.setLabel("bottom", "Time")
        self.flow_plot.setClipToView(True)
        self.flow_plot.getViewBox().setXLink(self.pressure_plot)
        self.pressure_pen = pg.mkPen(color="r", width=2)
        self.flow_pen = pg.mkPen(color="b", width=2)
        
        
        widget = QWidget()
        layout = QHBoxLayout()
        layout.addLayout(self.info_layout)
        layout.addWidget(self.display)
        widget.setLayout(layout)
        
        self.setCentralWidget(widget)
        
        # File IO
        self.File = self.menuBar().addMenu("&File")
        load_breaths = QAction("Load Sample", self)
        load_breaths.triggered.connect(self.__load_sample)
        self.File.addAction(load_breaths)
    
    def __next_breath(self):
        if self.current_breath < self.nbreaths-1:
            self.current_breath += 1
            self.breath_start = self.breaths["breath_start"].iat[self.current_breath]
            self.breath_end = self.breaths["breath_end"].iat[self.current_breath]
            self.pressure_plot.clear()
            self.pressure_plot.plot(np.linspace(0,self.data.iat[self.breath_end,0]-self.data.iat[self.breath_start,0], self.breath_end - self.breath_start), self.data.iloc[self.breath_start:self.breath_end,1].values, pen=self.pressure_pen)
            self.flow_plot.clear()
            self.flow_plot.plot(np.linspace(0,self.data.iat[self.breath_end,0]-self.data.iat[self.breath_start,0], self.breath_end - self.breath_start), self.data.iloc[self.breath_start:self.breath_end,2].values, pen=self.flow_pen)
            self.cur_breath_display.setText("Current breath is {} of {}".format(self.current_breath+1,self.nbreaths))
            self.cur_breath_label.setText("Current breath label is {}".format(self.labeled_breaths["target"].iat[self.current_breath]))
    
    def __prev_breath(self):
        if self.current_breath > 0:
            self.current_breath -= 1
            self.breath_start = self.breaths["breath_start"].iat[self.current_breath]
            self.breath_end = self.breaths["breath_end"].iat[self.current_breath]
            self.pressure_plot.clear()
            self.pressure_plot.plot(np.linspace(0,self.data.iat[self.breath_end,0]-self.data.iat[self.breath_start,0], self.breath_end - self.breath_start), self.data.iloc[self.breath_start:self.breath_end,1].values, pen=self.pressure_pen)
            self.flow_plot.clear()
            self.flow_plot.plot(np.linspace(0,self.data.iat[self.breath_end,0]-self.data.iat[self.breath_start,0], self.breath_end - self.breath_start), self.data.iloc[self.breath_start:self.breath_end,2].values, pen=self.flow_pen)
            self.cur_breath_display.setText("Current breath is {} of {}".format(self.current_breath+1,self.nbreaths))
            self.cur_breath_label.setText("Current breath label is {}".format(self.labeled_breaths["target"].iat[self.current_breath]))
    
    def __load_sample(self):
        sample_settings = SampleSelectionDialog()
        if sample_settings.exec():
            p, f = os.path.split(sample_settings.waveformFileName)
            if not os.path.isdir(os.path.join(p, "labeled_breaths")):
                os.mkdir(os.path.join(p,"labeled_breaths"))
            self.output_folder = os.path.join(p,"labeled_breaths")
            self.data = pd.read_csv(sample_settings.waveformFileName, usecols=sample_settings.waveformColumns)
            self.breaths = pd.read_csv(sample_settings.breathsFileName)
            self.states = pd.read_csv(sample_settings.statesFileName)
            if sample_settings.prevIndexFileName is not None:
                self.labeled_breaths = pd.read_csv(sample_settings.prevIndexFileName)
                self.nbreaths = self.labeled_breaths.shape[0]
            else:
                self.nbreaths = sample_settings.nbreaths if sample_settings.nbreaths < self.breaths.shape[0] else self.breaths.shape[0]
                samples = self.breaths.iloc[:,15:].sample(n=self.nbreaths)
                self.labeled_breaths = pd.concat([samples.reset_index(drop=True),pd.Series([None] * self.nbreaths, name="target"), pd.Series(samples.index, name="original_index")],axis=1)
            f = open(sample_settings.labelsFileName, "r")
            self.labels = f.read().split("\n")[:-1]
            f.close()
            for l in self.labels:
                b = QPushButton(l)
                b.clicked.connect(lambda : self.__label_breath(self.sender().text()))
                self.info_layout.addWidget(b)
            self.current_breath = 0
            self.breath_start = self.breaths["breath_start"].iat[self.current_breath]
            self.breath_end = self.breaths["breath_end"].iat[self.current_breath]
            self.pressure_plot.setLabel("left", self.data.columns[1])
            self.pressure_plot.setLabel("bottom", self.data.columns[0])
            self.flow_plot.setLabel("left", self.data.columns[2])
            self.flow_plot.setLabel("bottom", self.data.columns[0])
            self.pressure_plot.clear()
            self.pressure_plot.plot(np.linspace(0,self.data.iat[self.breath_end,0]-self.data.iat[self.breath_start,0], self.breath_end - self.breath_start), self.data.iloc[self.breath_start:self.breath_end,1].values, pen=self.pressure_pen)
            self.flow_plot.clear()
            self.flow_plot.plot(np.linspace(0,self.data.iat[self.breath_end,0]-self.data.iat[self.breath_start,0], self.breath_end - self.breath_start), self.data.iloc[self.breath_start:self.breath_end,2].values, pen=self.flow_pen)
            self.cur_breath_display.setText("Current breath is {} of {}".format(self.current_breath+1,self.nbreaths))
            self.cur_breath_label.setText("Current breath label is {}".format(self.labeled_breaths["target"].iat[self.current_breath]))
    
    def __label_breath(self, label):
        self.labeled_breaths["target"].iat[self.current_breath] = label
        self.labeled_breaths.to_csv(os.path.join(self.output_folder,"index.csv"))
        output_waveform = pd.concat([self.data.iloc[self.breath_start:self.breath_end,1:].reset_index(drop=True),self.states.iloc[self.breath_start:self.breath_end,:].reset_index(drop=True)],axis=1)
        output_waveform.to_csv(os.path.join(self.output_folder, "{}.csv".format(self.current_breath)))
        self.__next_breath()
        if self.current_breath == self.nbreaths-1:
            Popup("Alert","You have reached the end of all the breaths")
        
class SampleSelectionDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Load sample files")
        self.resize(800,600)
        
        self.main_widget = QVBoxLayout()
        # Get waveform file
        self.waveformFile = QLabel("")
        self.waveformButton = QPushButton("Load Waveform File")
        self.waveformButton.clicked.connect(self.__select_waveform_file)
        
        # Get breaths file
        self.breathsFile = QLabel("")
        self.breathsButton = QPushButton("Load Breaths File")
        self.breathsButton.clicked.connect(self.__select_breaths_file)
        
        # Get states file
        self.statesFile = QLabel("")
        self.statesButton = QPushButton("Load States File")
        self.statesButton.clicked.connect(self.__select_states_file)
        
        # Get label file
        self.labelsFile = QLabel("")
        self.labelsButton = QPushButton("Load Labels")
        self.labelsButton.clicked.connect(self.__select_labels_file)
        
        # Number of breaths to sample
        self.nbreathsFile = QLabel("How many breaths would you like to label? (This will select a new sample of breaths)")
        self.nbreathsEdit = QLineEdit()
        
        # Load previous index file
        self.prevIndexFile = QLabel("If you have a previous index file you wish to continue, please load it here. (This will override breaths to sample)")
        self.prevIndexButton = QPushButton("Load Index")
        self.prevIndexButton.clicked.connect(self.__select_prev_index)
        self.prevIndexFileName = None
        
        # Accept reject
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self._accepted)
        self.buttonBox.rejected.connect(self.reject)
        
        self.main_widget.addWidget(self.waveformFile)
        self.main_widget.addWidget(self.waveformButton)
        self.main_widget.addWidget(self.breathsFile)
        self.main_widget.addWidget(self.breathsButton)
        self.main_widget.addWidget(self.statesFile)
        self.main_widget.addWidget(self.statesButton)
        self.main_widget.addWidget(self.labelsFile)
        self.main_widget.addWidget(self.labelsButton)
        self.main_widget.addWidget(self.nbreathsFile)
        self.main_widget.addWidget(self.nbreathsEdit)
        self.main_widget.addWidget(self.prevIndexFile)
        self.main_widget.addWidget(self.prevIndexButton)
        self.main_widget.addWidget(self.buttonBox)
        
        self.setLayout(self.main_widget)
        
    def __select_waveform_file(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", os.getcwd(), "Tabular data (*.csv *.tsv *txt)")
        if fname[0] != "":
            data = pd.read_csv(fname[0], nrows=10)
            column_selection = ColumnSelectionDialog(data)
            if column_selection.exec():
                self.waveformFile.setText(fname[0])
                self.waveformFileName = fname[0]
                self.waveformColumns = column_selection.selected_indices
                
    def __select_breaths_file(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", os.getcwd(), "Tabular data (*.csv *.tsv *txt)")
        if fname[0] != "":
            self.breathsFileName = fname[0]
            self.breathsFile.setText(fname[0])
            data = pd.read_csv(fname[0])
            self.nbreathsFile.setText("There are {} breaths in this file. How many breaths would you like to label?".format(data.shape[0]))
            
    def __select_states_file(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", os.getcwd(), "Tabular data (*.csv *.tsv *txt)")
        if fname[0] != "":
            self.statesFileName = fname[0]
            self.statesFile.setText(fname[0])
    
    def __select_labels_file(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", os.getcwd(), "Tabular data (*.csv *.tsv *txt)")
        if fname[0] != "":
            self.labelsFileName = fname[0]
            self.labelsFile.setText(fname[0])
    
    def __select_prev_index(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", os.getcwd(), "Tabular data (*.csv *.tsv *.txt)")
        if fname[0] != "":
            self.prevIndexFileName = fname[0]
            self.prevIndexFile.setText(fname[0])
            
    def _accepted(self):
        if self.waveformFileName is not None and self.breathsFileName is not None and self.statesFileName is not None and self.labelsFileName is not None:
            if self.prevIndexFileName is not None:
                self.accept()
            elif self.nbreathsEdit.text().isnumeric():
                self.nbreaths = int(float(self.nbreathsEdit.text()))
                if self.nbreaths > 0:
                    self.accept()
                else:
                    Popup("Error", "Window length must be larger than 0")
            else:
                Popup("Error", "Window length must be numeric")
        else:
            Popup("Error", "Must specify all file inputs")
                
class ChangePointAnnotatorWindow(QMainWindow):
    pressure_options = [ps.peep.name, ps.pressure_rise.name, ps.pip.name, ps.pressure_drop.name]
    flow_options = [fs.no_flow.name, fs.inspiration_initiation.name, fs.peak_inspiratory_flow.name,
                    fs.inspiration_termination.name, fs.expiration_initiation.name, fs.peak_expiratory_flow.name,
                    fs.expiration_termination.name]
    pressure_pens = {ps.peep.name : pg.mkPen(color="g", width=2), ps.pressure_rise.name : pg.mkPen(color="b", width=2),
                     ps.pip.name : pg.mkPen(color="m", width=2), ps.pressure_drop.name : pg.mkPen(color="k", width=2)}
    flow_pens = {fs.no_flow.name : pg.mkPen(color="r", width=2), fs.inspiration_initiation.name : pg.mkPen(color="g", width=2),
                 fs.peak_inspiratory_flow.name : pg.mkPen(color="c", width=2), fs.inspiration_termination.name : pg.mkPen(color="m", width=2),
                 fs.expiration_initiation.name : pg.mkPen(color="y", width=2), fs.peak_expiratory_flow.name : pg.mkPen(color="k", width=2),
                 fs.expiration_termination.name : pg.mkPen(color="b", width=2)}
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Neonatal Ventilation Data Annotation Tool")
        self.resize(1280,720)
        pg.setConfigOption('foreground', "k")
        pg.setConfigOption('background', 'w')
        
        # Label choices
        info_layout = QHBoxLayout()
        self.pressure_choice = QComboBox()
        self.pressure_choice.addItems(ChangePointAnnotatorWindow.pressure_options)
        self.flow_choice = QComboBox()
        self.flow_choice.addItems(ChangePointAnnotatorWindow.flow_options)
        info_layout.addWidget(QLabel("Pressure labels"), 1)
        info_layout.addWidget(self.pressure_choice, 2)
        info_layout.addWidget(QLabel("Flow labels"), 1)
        info_layout.addWidget(self.flow_choice, 2)
        
        # Graphing
        self.display = pg.GraphicsView()
        self.display_layout = pg.GraphicsLayout()
        self.display.setCentralItem(self.display_layout)
        self.pressure_plot = self.display_layout.addPlot(title="Pressure waveform")
        self.pressure_plot.showGrid(x=True, y=True)
        self.pressure_plot.setMenuEnabled(False)
        self.pressure_plot.setLabel("left", "Pressure")
        self.pressure_plot.setLabel("bottom", "Time")
        self.pressure_plot.scene().sigMouseClicked.connect(self.__on_graph_click)
        self.pressure_plot.sigXRangeChanged.connect(self.__on_axis_change)
        self.pressure_plot.setClipToView(True)
        self.display_layout.nextRow()
        self.flow_plot = self.display_layout.addPlot(title="Flow waveform")
        self.flow_plot.showGrid(x=True, y=True)
        self.flow_plot.setMenuEnabled(False)
        self.flow_plot.setLabel("left", "Flow")
        self.flow_plot.setLabel("bottom", "Time")
        self.flow_plot.setClipToView(True)
        self.flow_plot.getViewBox().setXLink(self.pressure_plot)
        self.pressure_pen = pg.mkPen(color="r", width=2)
        self.flow_pen = pg.mkPen(color="b", width=2)
        
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addLayout(info_layout)
        layout.addWidget(self.display)
        widget.setLayout(layout)
        
        self.setCentralWidget(widget)
        
        # Default values
        self.data = None
        self.w_len = 500
        self.interval = 0
        self.pressure_labels = []
        self.flow_labels = []
        self.pressure_markers = []
        self.flow_markers = []
        self.xmin = 0
        self.xmax = 0
        
        # For file i/o
        self.File = self.menuBar().addMenu("&File")
        load_waveforms = QAction("Load Waveforms", self)
        load_waveforms.triggered.connect(self.__load_waveforms)
        self.File.addAction(load_waveforms)
        save_annotations = QAction("Save Annotations", self)
        save_annotations.triggered.connect(self.__save_annotations)
        self.File.addAction(save_annotations)
        load_annotations = QAction("Load Annotations", self)
        load_annotations.triggered.connect(self.__load_annotations)
        self.File.addAction(load_annotations)
        save_graph = QAction("Save graph", self)
        save_graph.triggered.connect(self.__save_graph)
        self.File.addAction(save_graph)
        
        # For view settings
        self.View = self.menuBar().addMenu("&View")
        view_settings = QAction("Display Settings", self)
        view_settings.triggered.connect(self.__display_setting)
        self.View.addAction(view_settings)
        
    def __load_waveforms(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", os.getcwd(), "Tabular data (*.csv *.tsv *txt)")
        if fname[0] != "":
            data = pd.read_csv(fname[0], nrows=100)
            column_selection = ColumnSelectionDialog(data)
            if column_selection.exec():
                self.data = pd.read_csv(fname[0], usecols=column_selection.selected_indices)
                self.data[self.data.columns[0]] -= self.data[self.data.columns[0]].min()
                self.data[self.data.columns[0]] = np.linspace(0,self.data[self.data.columns[0]].max(), self.data.shape[0])
                self.pressure_labels = np.array([-1] * self.data.shape[0])
                self.flow_labels = np.array([-1] * self.data.shape[0])
                self.pressure_markers = {}
                self.flow_markers = {}
                self.interval = (self.data.iat[-1,0] - self.data.iat[0,0]) / self.data.shape[0]
                self.__init_graph()
                
    def __load_annotations(self):
        if self.data is not None:
            fname = QFileDialog.getOpenFileName(self, "Open File", os.getcwd(), "(*.csv)")
            if fname[0] != "":
                labels = pd.read_csv(fname[0])
                self.pressure_labels = np.array([-1] * self.data.shape[0])
                self.flow_labels = np.array([-1] * self.data.shape[0])
                for i in range(labels.shape[0]):
                    self.pressure_labels[labels.iat[i,0]] = labels.iat[i,1]
                    self.flow_labels[labels.iat[i,0]] = labels.iat[i,2]
                self.__clear_markers()
                self.__plot_markers(self.xmin, self.xmax)
        else:
            Popup("Error", "No data loaded so labels cannot be loaded")
        
    def __save_annotations(self):
        if self.data is not None:
            fname = QFileDialog.getSaveFileName(self, "Save File", os.getcwd(), "(*.csv)")
            if fname[0] != "":
                output = []
                for i in range(len(self.pressure_labels)):
                    if self.pressure_labels[i] != -1 or self.flow_labels[i] != -1:
                        row = {}
                        row["index"] = i
                        row["pressure_annotations"] = self.pressure_labels[i]
                        row["flow_annotations"] = self.flow_labels[i]
                        output += [row]
                output = pd.DataFrame(output)
                if fname[0].find(".csv") > -1:
                    output.to_csv(fname[0], index=False)
                else:
                    output.to_csv(fname[0] + ".csv", index=False)
        else:
            Popup("Error", "No data loaded so there cannot be any labels")
    
    def __save_graph(self):
        fname = QFileDialog.getSaveFileName(self, "Save File", os.getcwd(), "(*.jpg *.png)")
        if fname[0] != "":
            exporter = pg.exporters.ImageExporter(self.display.scene())
            exporter.export(fname[0])
    
    def __display_setting(self):
        display_setting = DisplaySettingsDialog(self.w_len)
        if display_setting.exec():
            self.w_len = display_setting.w_len
            if self.data is not None:
                self.flow_plot.setXRange(self.flow_plot.viewRange()[0][0], self.flow_plot.viewRange()[0][0] + self.w_len * self.interval)
            
    def __init_graph(self):
        self.__clear_markers()
        self.pressure_plot.setLabel("left", self.data.columns[1])
        self.pressure_plot.setLabel("bottom", self.data.columns[0])
        self.flow_plot.setLabel("left", self.data.columns[2])
        self.flow_plot.setLabel("bottom", self.data.columns[0])
        self.pressure_plot.plot(self.data.iloc[:,0], self.data.iloc[:,1], pen=self.pressure_pen)
        self.flow_plot.plot(self.data.iloc[:,0], self.data.iloc[:,2], pen=self.flow_pen)
        self.flow_plot.setXRange(0,self.w_len * self.interval)
        self.xmin = 0
        self.xmax = int(self.w_len * self.interval)
    
    def __on_axis_change(self, event):
        lower = event.viewRange()[0][0]
        if lower < 0:
            lower = 0
        upper = event.viewRange()[0][1]
        if upper > self.data.shape[0] * self.interval:
            upper = self.data.shape[0] * self.interval
        temp = {}
        for key in self.pressure_markers.keys():
            if key not in range(int(lower / self.interval), int(upper / self.interval)):
                self.pressure_plot.removeItem(self.pressure_markers[key])
            else:
                temp[key] = self.pressure_markers[key]
        self.pressure_markers = temp
        temp = {}
        for key in self.flow_markers.keys():
            if key not in range(int(lower / self.interval), int(upper / self.interval)):
                self.flow_plot.removeItem(self.flow_markers[key])
            else:
                temp[key] = self.flow_markers[key]
        self.flow_markers = temp
        if upper > self.xmax:
            self.__plot_markers(self.xmax, upper)
        elif lower < self.xmin:
            self.__plot_markers(lower, self.xmin)
        self.xmin = lower
        self.xmax = upper
    
    def __on_graph_click(self, event):
        if self.data is None:
            return
        if self.pressure_plot.getAxis("bottom") == event.currentItem:
            xpos = self.pressure_plot.getViewBox().mapSceneToView(event.scenePos()).x()
            xind = int(xpos / self.interval)
            if xind < 0 or xind >= self.data.shape[0]:
                return
            if self.pressure_labels[xind] == ps[self.pressure_choice.currentText()].value:
                self.pressure_labels[xind] = -1
                if xind in self.pressure_markers.keys():
                    self.pressure_plot.removeItem(self.pressure_markers[xind])
                    del(self.pressure_markers[xind])
            else:
                self.pressure_labels[xind] = ps[self.pressure_choice.currentText()].value
                if xind in self.pressure_markers.keys():
                    self.pressure_plot.removeItem(self.pressure_markers[xind])
                    del(self.pressure_markers[xind])
                self.pressure_markers[xind] = self.pressure_plot.addLine(x=self.data.iat[xind,0], pen=ChangePointAnnotatorWindow.pressure_pens[self.pressure_choice.currentText()], label=self.pressure_choice.currentText(), labelOpts={"position" :(ps(self.pressure_labels[xind]).value+1)/5})
            
        elif self.flow_plot.getAxis("bottom") == event.currentItem:
            xpos = self.pressure_plot.getViewBox().mapSceneToView(event.scenePos()).x()
            xind = int(xpos / self.interval)
            if xind < 0 or xind >= self.data.shape[0]:
                return
            if self.flow_labels[xind] == fs[self.flow_choice.currentText()].value:
                self.flow_labels[xind] = -1
                if xind in self.flow_markers.keys():
                    self.flow_plot.removeItem(self.flow_markers[xind])
                    del(self.flow_markers[xind])
            else:
                self.flow_labels[xind] = fs[self.flow_choice.currentText()].value
                if xind in self.flow_markers.keys():
                    self.flow_plot.removeItem(self.flow_markers[xind])
                    del(self.flow_markers[xind])
                self.flow_markers[xind] = self.flow_plot.addLine(x=self.data.iat[xind,0], pen=ChangePointAnnotatorWindow.flow_pens[self.flow_choice.currentText()], label=self.flow_choice.currentText(), labelOpts={"position" : (fs(self.flow_labels[xind]).value + 1)/8})
    
    def __clear_markers(self):
        for k,v in self.pressure_markers.items():
            self.pressure_plot.removeItem(v)
        for k,v in self.flow_markers.items():
            self.flow_plot.removeItem(v)
        self.pressure_markers = {}
        self.flow_markers={}
    
    def __plot_markers(self, xmin,xmax):
        for i in range(int(xmin / self.interval), int(xmax / self.interval)):
            if self.pressure_labels[i] != -1:
                self.pressure_markers[i] = self.pressure_plot.addLine(self.data.iat[i,0], pen=self.pressure_pens[ps(self.pressure_labels[i]).name], label=ps(self.pressure_labels[i]).name, labelOpts={"position" :(ps(self.pressure_labels[i]).value+1)/5})
            if self.flow_labels[i] != -1:
                self.flow_markers[i] = self.flow_plot.addLine(self.data.iat[i,0], pen=self.flow_pens[fs(self.flow_labels[i]).name], label=fs(self.flow_labels[i]).name,labelOpts={"position" : (fs(self.flow_labels[i]).value + 1)/8})

class DisplaySettingsDialog(QDialog):
    def __init__(self, w_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_len = w_len
        self.setWindowTitle("Set display settings")
        
        w_len_row = QHBoxLayout()
        w_len_row.addWidget(QLabel("Window Length"))
        self.w_len_entry = QLineEdit()
        self.w_len_entry.setText(str(self.w_len))
        w_len_row.addWidget(self.w_len_entry)
        
        layout = QVBoxLayout()
        layout.addLayout(w_len_row)
        
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self._accepted)
        self.buttonBox.rejected.connect(self.reject)
        
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)
        
    def _accepted(self):
        if self.w_len_entry.text().isnumeric():
            self.w_len = int(float(self.w_len_entry.text()))
            if self.w_len > 0:
                self.accept()
            else:
                Popup("Error", "Window length must be larger than 0")
        else:
            Popup("Error", "Window length must be numeric")
        

class ColumnSelectionDialog(QDialog):
    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Select the columns that contain Time, Pressure, and Flow")
        self.resize(800,600)
        
        self.data = data
        self.model = TableModel(self.data)
        self.tableview = QTableView()
        self.tableview.setModel(self.model)
        
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self._accepted)
        self.buttonBox.rejected.connect(self.reject)
        
        layout = QVBoxLayout()
        layout.addWidget(self.tableview)
        layout.addWidget(self.buttonBox)
        
        self.setLayout(layout)
    
    def _accepted(self):
        indexes = self.tableview.selectionModel().selectedColumns()
        if len(indexes) == 3:
            self.selected_indices = []
            for i in sorted(indexes):
                self.selected_indices.append(i.column())
            if self.data.iloc[:,0].dtype == "int64":
                self.accept()
            else:
                Popup("Error","Time column must be of type integer")
        else:
            Popup("Error", "Please select columns for Time, Pressure, and Flow")

class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self.data = data
    def data(self, index, role):
        if role == Qt.DisplayRole:
            return str(self.data.iat[index.row(), index.column()])
    def rowCount(self, index):
        return self.data.shape[0]
    def columnCount(self, index):
        return self.data.shape[1]
    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self.data.columns[section])
            elif orientation == Qt.Vertical:
                return str(self.data.index[section])


class Popup(QDialog):
    def __init__(self, header, msg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle(header)
        layout = QVBoxLayout()
        layout.addWidget(QLabel(msg))
        self.setLayout(layout)
        self.exec()
