import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ventiliser",
    version="0.0.1",
    author="David Chong Tian Wei",
    author_email="dtwc3@cam.ac.uk",
    description="Provides a pipeline for segmenting pressure and flow data from ventilators into individual breaths with an attempt to identify sub-phases. Developed on data from neonates ventilated on Draeger ventilators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/barrinalo/ventiliser",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
