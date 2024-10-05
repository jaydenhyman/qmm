# Qualitative Mathematical Modelling (QMM) in Python

![QMM Logo](https://github.com/jaydenhyman/qmm/blob/1d58b3b48173f534eaa535b9c891d159a463da6e/logo.png)

Qualitative Mathematical Modelling (QMM) is a method for analysing the structure of complex systems where general causal relationships are known but precise quantitative data are lacking. By utilising networks (signed digraphs) to visualise system structure, QMM enables users to build models of real-world systems and generate scientifically testable predictions of system response to perturbations. As an open-source software tool, QMM provides insights into system behaviour for researchers and practitioners across various fields, including ecology, natural resource management, epidemiology, economics, and the social sciences.

## Features

- Python package (`qmm`) for qualitative mathematical modelling, including core modules for defining model structure, stability analysis, perturbation analysis and making qualitative predictions.
- Integration with an interactive web application for creating signed digraph (network) models representing the mathematical structure of a complex system.

## Contact

For any additional information or questions, please contact:

Jayden Hyman: <j.hyman@uq.edu.au>

## How to use

1. Install Python and required packages:

   a. Install Python 3.10

      Download and install Python 3.10 from <https://www.python.org/downloads/>

   b. Install Required Packages Using pip

      Open a command prompt or terminal window and run:

      ```bash
      pip install numpy==1.26.4 networkx==3.3 pandas==2.0.2 numba==0.60.0 sympy==1.13 seaborn==0.13.2 matplotlib==3.9.2
      ```

   c. Install JupyterLab

      ```bash
      pip install jupyterlab
      ```

2. Use the web application to create and save models: [Open in browser](https://d2x70551if0frn.cloudfront.net/). An example model file (`test/mesocosm.json`) is provided.

3. The `qmm.ipynb` file provides core functions to analyse model files (e.g. `test/mesocosm.json`).

4. Open Jupyter:

   After installation, open a command prompt or terminal window and run:

   ```bash
   jupyter lab
   ```

   This will launch JupyterLab in your default web browser, where you can navigate to and open the `qmm.ipynb` file.

## Documentation

Detailed documentation for the `qmm` package and its modules is not currently available.

## Licensing

This model is licensed under a BSD 3-Clause License. See LICENSE.md for further information.

## Attribution

A Zenodo will be available in the near future for attribution.

## Contributing

We welcome contributions to improve and expand the QMM software. As the project is in its early stages of development, we appreciate your patience and support in helping us refine the software.
