# Simulation code and analysis for the paper "Transition from traveling fronts to diffusion-limited growth in expanding populations".

The paper simulates and analyses the results of the following coupled PDEs:
$$
\frac{\partial b}{\partial t} = D_b\bm{\nabla}\cdot(bn\bm{\nabla} b) + \gamma bn
$$
$$
\frac{\partial n}{\partial t} = D_n\bm{\nabla}^2 n - \gamma bn
$$

- NLDv3.py and NLD2d.py contain the classes and functions for simulating the PDEs in one and two dimensions respectively
- analysis.ipynb is a notebook that contains all the analysis and the code to produce the figures of the paper
