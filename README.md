# tjml-q2project

# Time-Optimized Bayesian Optimization of Random Forest Hyperparameters

**Authors:** Gabriel Xu and Andrew Chen  
**Date:** January 31, 2025  
**Project:** Machine Learning Quarter 2 Project  
**Advisor:** Dr. Yilmaz

## Overview

This repository contains code, data, and documentation for our research project on optimizing hyperparameters in Random Forest classifiers. We use a meta-optimization approach where Bayesian Optimization is used to tune its own hyperparameters, reducing computational time while maintaining performance.

## Files in the Repository

- **Research Documents:**
  - `Revised Q2 Project Report.pdf` – Final research paper.
  - `Revised Q2 Project Report.docx` – Source document for the research paper.
  - `Q2 Presentation.pdf` & `Q2 Presentation.pptx` – Presentation files.

- **Source Code:**
  - `bayesian-optimization.py` – Implementation of Bayesian Optimization-based tuning.
  - `no-optimization.py` – Baseline Random Forest with manual hyperparameters.
  - `optimized.py` – Meta-optimized Bayesian Optimization code.

- **Datasets:**
  - `breast-cancer.csv` – Primary dataset from Weka.
  - `train.csv` & `test.csv` – Training and testing data splits.

## Project Details

- **Problem:** Manually tuning hyperparameters is inefficient. Our project uses Bayesian Optimization for this task and further optimizes its settings automatically.
- **Methods:** We compare three methods:
  1. Baseline Random Forest with manual tuning.
  2. Random Forest tuned with standard Bayesian Optimization.
  3. Meta-optimized Bayesian Optimization that tunes its own hyperparameters.
- **Results:** Our meta-optimized approach achieves similar accuracy and precision to standard Bayesian Optimization but with a significant reduction in execution time.

## How to Run

1. **Install Dependencies:**  
   Ensure you have Python 3.x and install required libraries (e.g., scikit-learn, numpy, pandas, matplotlib).  
   Create a `requirements.txt` if needed, then run:  
   ```bash
   pip install -r requirements.txt
