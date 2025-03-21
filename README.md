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

- **Abstract:**
- Finding optimal hyperparameters is crucial for machine learning models, yet it is both challenging and time-consuming. Traditional approaches, such as manual hyperparameter selection and grid search, rely heavily on trial and error and often produce inconsistent results. Bayesian Optimization has been proposed as a more efficient alternative, which leverages a probabilistic surrogate model to help guide hyperparameter selection. Nevertheless, existing approaches rely on the manual selection of Bayesian Optimization's hyperparameters itself, and such selection can introduce bias and cause inefficiencies. This study presents a novel approach that meta-optimizes the Bayesian Optimization process, allowing it to tune its own hyperparameters while optimizing the Random Forest Classifier. We apply this method to the breast-cancer.csv dataset from Weka. We compare three models: (1) a baseline Random Forest with manually chosen hyperparameters, (2) a Bayesian-optimized Random Forest with predefined settings, and (3) our meta-optimized approach, where Bayesian Optimization itself is tuned recursively. Performance is evaluated based on classification accuracy, precision, recall, training time, and confusion matrices. Preliminary results indicate that meta-optimization improves classification accuracy and reduces hyperparameter search time. Our findings suggest that automating Bayesian Optimization settings can enhance the performance and efficiency of machine learning classifiers. Future work may explore applying this approach to other machine learning models and different datasets.

## How to Run

1. **Install Dependencies:**  
   Ensure you have Python 3.x and install required libraries (e.g., scikit-learn, numpy, pandas, matplotlib).  
   Create a `requirements.txt` if needed, then run:  
   ```bash
   pip install -r requirements.txt
