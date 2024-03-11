# LODED: Linearised ODE Designs

## Overview

LODED is a research project developed as part of a Master thesis, focusing on the Optimal Experimental Design (OED) for obtaining sampling schemes for estimation of parameters of non-linear Ordinary Differential Equations (ODEs). Inspired by classical fields of OED and operator theoretic approach to dynamical systems, this project uses deep learning to find linear representations of dynamical systems and classical OED to find experimental designs. 

## Key Concepts

- **Linear Experimental Design for linear ODEs**: Utilizes statistical techniques to optimize the allocation of experimental resources. The goal is to maximize the information gained about a system's parameters while minimizing experimental costs and efforts.

- **Linear representation of Dynamical Systems**: A transformative approach that applies neural network embeddings to represent non-linear dynamical systems in a linearized form. This enables the use of linear OED methods for systems that are non-linear.

- **Linearised ODE Designs (LODED)**: Combines the ideas for linear representations and linear experimental design to find designs that for the linearised dynamical system. 


![](docs/images/Overview.png)


## Installation

To install LODED, follow these steps:

```bash
# Clone the repository
git clone https://github.com/AntanasMurelis/LODED.git
cd LODED

# It's recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # Use `venv\Scripts\activate` on Windows

# Install the project
pip install .

# Alternatively, for development purposes, you might want to install the project in editable mode
pip install -e .

