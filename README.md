# PGM-based digital-twin

Pruning Bayesian Networks for Computationally Tractable Multi-Model Calibration

## Description

Anomaly response in aerospace systems increasingly relies on multi-model analysis in digital twins to replicate the system's behaviors and inform decisions. However, computer model calibration methods are typically deployed on individual models and are limited in their ability to capture dependencies across models. In addition, model heterogeneity has been a significant issue in integration efforts. Bayesian Networks are well suited for multi-model calibration tasks as they can be used to formulate a mathematical abstraction of model components and encode their relationship in a probabilistic and interpretable manner. The computational cost of this method however increases exponentially with the graph complexity. In this work, we propose a graph pruning algorithm to reduce computational cost while minimizing the loss in calibration ability by incorporating domain-driven metrics for selection purposes. We implement this method using a Python wrapper for BayesFusion software and show that the resulting prediction accuracy outperforms existing pruning approaches which rely primarily on statistics.

## Dependencies

This project relies on the following Python packages:

- matplotlib
- numpy


You can install these dependencies using pip:

```bash
pip install matplotlib numpy
```

The project also requires installing the GeNIe Modeler and the Python API for the SMILE engine accessible on the *[BayesFusion website](https://www.bayesfusion.com/)*.

## Execution

Run the Python script: `main.py`

## Authors

* [Nicolas Gratius](https://www.linkedin.com/in/nicolas-gratius-3360b0110/)

* [Mario Bergés](https://www.cmu.edu/cee/people/faculty/berges.html)

* [Burcu Akinci](https://www.cmu.edu/cee/people/faculty/akinci.html)

## Acknowledgments

[NASA grant 80NSSC19K1052](https://govtribe.com/award/federal-grant-award/grant-for-research-80nssc19k1052)
[BayesFusion academic license](https://www.bayesfusion.com/downloads/)