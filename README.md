# Applying Threshold-Free Cluster Enhancement (TFCE) to EEG/ERP Experimental Designs

### A Comprehensive, Reproducible Tutorial with MATLAB and R Implementations

[![https://www.linkedin.com/in/yufang-w-1295881b5/](https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555)](https://www.linkedin.com/in/yufang-w-1295881b5/) [![https://github.com/Yufanggg](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white&colorB=555)](https://github.com/Yufanggg) <img alt="GitHub" src="https://img.shields.io/github/license/bopith/UnicornCompanies?style=for-the-badge"> 

**Author:** Yufang Wang

[LinkedIn](https://www.linkedin.com/in/yufang-w-1295881b5/) •
[GitHub](https://github.com/Yufanggg)

---

## Overview

Threshold-Free Cluster Enhancement (TFCE) has become one of the most widely used methods for multiple-comparison correction in neuroimaging because it avoids the need to specify an arbitrary cluster-forming threshold while maintaining high statistical sensitivity.

Although TFCE has been widely applied to EEG/ERP studies, practical guidance on adapting TFCE to different experimental designs remains limited. Different experimental designs require different permutation schemes to satisfy the exchangeability assumption, and applying an inappropriate permutation strategy may lead to invalid statistical inference.

This repository accompanies the tutorial

> **Applying Threshold-Free Cluster Enhancement (TFCE) to EEG/ERP Experimental Designs: A Comprehensive Tutorial**

and provides fully reproducible implementations of TFCE analyses in **MATLAB** and **R**, together with simulated EEG datasets covering a broad range of commonly used experimental designs.

---

## Features

- Complete TFCE analysis workflow
- Fully reproducible simulated EEG/ERP datasets
- MATLAB implementations
- R implementations
- Eight representative EEG/ERP experimental designs
- Appropriate permutation strategies for each design
- Publication-quality figures reproduced from the tutorial

---

## Experimental Designs

The tutorial covers the following experimental designs.

| Tutorial | Experimental Design | Permutation Strategy |
|------------|--------------------|----------------------|
| TFCE01 | Between-subject | Label permutation |
| TFCE02 | Between-subject with covariates | Freedman–Lane permutation |
| TFCE03 | 2 × 2 between-subject factorial | Restricted permutation |
| TFCE04 | 2 × 2 interaction | Freedman–Lane permutation |
| TFCE05 | Within-subject | Sign flipping |
| TFCE06 | Mixed (split-plot) design | Restricted permutation |
| TFCE07 | Nested design (e.g., students within classes) | Block permutation |
| TFCE08 | Fully crossed subject–item design | Within subject–item permutation |

---

## Repository Structure

```
TFCE_Tutorial_DOE
│
├── MATLAB/        MATLAB implementations
├── R/             R implementations
├── DataSim/       Simulation scripts
├── Data/          Simulated EEG datasets
├── Figures/       Figures used in the tutorial
├── Results/       Example TFCE outputs
├── paper/         Tutorial manuscript
└── README.md
```


---

# Simulated Data

The repository contains simulated EEG datasets for every experimental design presented in the tutorial.

Each dataset includes

- raw EEG signals
- experimental design table
- ground-truth effects
- visualization scripts

The simulated datasets are intended for

- learning TFCE
- benchmarking implementations
- validating new methods
- teaching permutation testing

---

## Example 1: Between-Subject Design

Ground truth:

*(Insert Figure 1 here)*

The simulated ERP contains a P300 effect for the treatment condition.

Additional visualizations include

- simulated ERP waveforms
- scalp topography
- statistical maps
- TFCE results

---

## Example 2: 2 × 2 Factorial Design

Ground truth:

*(Insert Figure 2 here)*

The simulated dataset contains

- main effect of Factor A
- main effect of Factor B
- interaction effect

with corresponding TFCE results.

---

# MATLAB Implementation

The `MATLAB/` directory contains complete implementations for every experimental design.

Example:

```matlab
TFCE01_betweenSubject_simple.m
```

Each script

- loads the simulated data
- performs permutation testing
- computes TFCE statistics
- estimates corrected p-values
- generates publication-quality figures

### Required Toolboxes

- Statistics and Machine Learning Toolbox

No additional proprietary software is required.

---

# R Implementation

The `R/` directory contains equivalent implementations written in R.

Example

```r
source("R/TFCE01_betweenSubject_simple.R")
```

### Required Packages

Typical packages include

- lme4
- lmerTest
- tidyverse
- permuco
- Matrix

Install packages with

```r
install.packages(c(
  "lme4",
  "lmerTest",
  "tidyverse",
  "permuco"
))
```

---

# Getting Started

Clone the repository

```bash
git clone https://github.com/Yufanggg/TFCE_Tutorial_DOE.git
```

Run one of the tutorials

MATLAB

```matlab
TFCE01_betweenSubject_simple.m
```

or

R

```r
source("R/TFCE01_betweenSubject_simple.R")
```

---

# Reproducing the Tutorial

The repository is organised so that every example in the tutorial can be reproduced directly.

Workflow

```
Simulated EEG Data
        │
        ▼
Permutation Testing
        │
        ▼
TFCE Statistic
        │
        ▼
Corrected p-values
        │
        ▼
Visualization
```

---

# Citation

If you use this repository in your research, please cite

> Wang, Y. *Applying Threshold-Free Cluster Enhancement (TFCE) to EEG/ERP Experimental Designs: A Comprehensive Tutorial.*

(BibTeX will be added after publication.)

---

# License

This project is released under the MIT License.

---
