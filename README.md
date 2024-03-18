# Results

This repository contains results obtained with [`sbibm`](https://github.com/sbi-benchmark/sbibm), a simulation-based inference benchmark, as well as the scripts and instructions to replicate them.


## Contents

Folder             | Description
------------------ | -----------
[`benchmarking_sbi`](https://github.com/sbi-benchmark/results/tree/main/benchmarking_sbi) | Results and code for `Benchmarking Simulation-Based Inference`


This code was used to train the posterior estimates for the benchmarking experiment in [L-C2ST: Local Diagnostics for Posterior Approximations in Simulation-Based Inference](https://arxiv.org/pdf/2306.03580). 

It requires to clone and modifie the [`sbibm`](https://github.com/sbi-benchmark/sbibm) repository. The modified files are in the `sbibm_changes` folder:
- `snpe.py`: needs to return the trained posterior estimator.
- `bernoulli_glm/task.py`: the reference_posterior sampler is missing the `observation` input variable.
