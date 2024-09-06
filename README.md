# QMM: Qualitative Mathematical Modelling in Python

![QMM Logo](https://github.com/jaydenhyman/qmm/blob/6f9beb2d90807e8fbc9622a4db3cdf7c0fcd06aa/logo.png)

QMM is an open-source Python package that offers a robust and accessible approach to analysing the structure and function of complex systems through an integrated web application and Python package. The integrated software program is a platform for building, analysing and presenting generalisable models of real-world systems and is an open-access resource for theoretical and applied science.

## Features

- Interactive web-based model building application for creating signed digraph models.
- Comprehensive Python package (`qmm`) for analysing qualitative mathematical models.
- Modules for defining model structure, stability analysis, press perturbation and making qualitative predictions.
- Uses well-established Python libraries including `networkx`, `sympy`, `numpy` and `pandas`.

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

   This method uses Anaconda's default packages, which include NumPy, SymPy, NetworkX, Pandas, Numba, and Graphviz. JupyterLab provides an integrated development environment for running the QMM package.

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

   Option 3: Using Python directly

   a. Install Python 3.10 from <https://www.python.org/>

   b. Install Graphviz 0.20.3 from <https://graphviz.org/download/>

   c. Open a command prompt and install the required packages:

      ```bash
      pip install numpy==1.26.4 networkx==3.3 pandas==2.0.2 numba==0.60.0 sympy==1.13 graphviz==0.20.3
      ```

   Note: If you encounter issues installing Graphviz, consider using the Anaconda or Miniconda method instead.

2. Use the web-based model building application to create signed digraph models: [Open in browser](https://d2x70551if0frn.cloudfront.net/)

3. The `qmm.ipynb` file provides core functions to analyse signed digraph models. To get started with analysing your model, open this file in JupyterLab or your preferred Python IDE.

## Documentation

Detailed documentation for the `qmm` package and its modules is not currently available.

## Licensing

This model is licensed under a BSD 3-Clause License. See LICENSE.md for further information.

## Attribution

A Zenodo will be available in the near future for attribution.

## Contributing

We welcome contributions to improve and expand the QMM software. As the project is in its early stages of development, we appreciate your patience and support in helping us refine the software.
