# Qualitative Mathematical Modelling (QMM) in Python

![QMM Logo](https://github.com/jaydenhyman/qmm/blob/1d58b3b48173f534eaa535b9c891d159a463da6e/logo.png)

Qualitative Mathematical Modelling (QMM) is a method for analysing the structure of complex systems where general causal relationships are known but precise quantitative data are lacking. By utilising signed digraphs to visualise network structure, QMM enables users to build models of real-world systems and generate scientifically robust and testable predictions of system response to perturbations. As an open-source software tool, QMM provides insights into system behaviour for researchers and practitioners across various fields, including ecology, natural resource management, epidemiology, economics, and the social sciences.

## Features

- Interactive web application for creating signed digraph (network) models representing the mathematical structure of a complex system.
- Python package (`qmm`) for qualitative mathematical modelling, including core modules for defining model structure, stability analysis, perturbation analysis and making qualitative predictions.

## Contact

For any additional information or questions, please contact:

Jayden Hyman: <j.hyman@uq.edu.au>

## How to use

1. Install Python and required packages:

   Option 1: Using Anaconda and JupyterLab (Recommended)

   a. Install Anaconda from <https://www.anaconda.com/products/distribution>

   b. Launch Anaconda Navigator

   c. Create a new environment:
      - Click on "Environments" in the left sidebar
      - Click "Create" at the bottom
      - Name your environment (e.g., "qmm") and select Python 3.10
      - Click "Create"
      - In the "Environments" list, click on your newly created environment

   d. Install JupyterLab:
      - With your new environment selected, go to the "Home" tab
      - Find JupyterLab in the list of applications
      - Click "Install"

   This method uses Anaconda's default packages, which include NumPy, SymPy, NetworkX, Pandas, Numba, and Graphviz. JupyterLab provides an integrated development environment for running QMM package functions.

   Option 2: Using Miniconda

   a. Install Miniconda from <https://docs.conda.io/en/latest/miniconda.html>

   b. Open a Miniconda prompt and create a new environment:

      ```bash
      conda create -n qmm python=3.10
      conda activate qmm
      ```

   c. Install the required packages:

      ```bash
      conda install numpy=1.26.4 networkx=3.3 pandas=2.0.2 numba=0.60.0 sympy=1.13
      conda install -c conda-forge graphviz=0.20.3
      ```

2. Use the web application to create signed digraph models: [Open in browser](https://d2x70551if0frn.cloudfront.net/)

3. The `qmm.ipynb` file provides core functions to analyse signed digraph models. To get started with analysing your model, open this file in JupyterLab or your preferred Python IDE.

## Documentation

Detailed documentation for the `qmm` package and its modules is not currently available.

## Licensing

This model is licensed under a BSD 3-Clause License. See LICENSE.md for further information.

## Attribution

A Zenodo will be available in the near future for attribution.

## Contributing

We welcome contributions to improve and expand the QMM software. As the project is in its early stages of development, we appreciate your patience and support in helping us refine the software.
