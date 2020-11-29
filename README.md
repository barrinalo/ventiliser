### Ventiliser

Ventiliser is a tool for segmenting pressure and flow waveforms from ventilators into breaths. It also attempts to find the position of breath sub-phases. Ventiliser was developed on data from neonates ventialted on Draeger ventilators.

### Installation
Ventiliser depends on numpy, pandas, scipy, atpbar, and PyQT5. Installation via pip is recommended.

```python
pip install ventiliser
```

### Usage

An example of a simple script can be seen below

```python
from ventiliser.GeneralPipeline import GeneralPipeline

pipeline = GeneralPipeline()
pipeline.configure() # For information on parameters you can configure see docs
pipeline.load_data("path to data", [0,1,2]) # [0,1,2] refers to the columns in your data file corresponding to time, pressure, flow
pipeline.process() # You can suppress log and output files by setting them false. See docs for more information

# To access the breaths programmatically you can do as follows
pipeline.labeller.get_breaths_raw()
# To access the pressure and state labels programmatically you can do as follows
pipeline.mapper.p_labels
pipeline.mapper.f_labels
```

Please view the docs for more detailed information on optional parameters.

### Contact

For questions please email

gbelteki@aol.com (Dr. Gusztav Belteki)

dtwc3@cam.ac.uk (David Chong)

### License

Copyright (c) 2020 David Chong Tian Wei, Gusztav Belteki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
