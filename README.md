# tjml-q2project

# Time-Optimized Bayesian Optimization of Random Forest Hyperparameters

# Time-Optimized Bayesian Optimization of Random Forest Hyperparameters

## Project Overview
This project implements a meta-optimized approach to Bayesian Optimization for tuning hyperparameters of a Random Forest Classifier. Our method recursively tunes the hyperparameters of the Bayesian Optimization algorithm itself, resulting in improved classification accuracy and reduced hyperparameter search time compared to conventional methods. The approach is demonstrated on the breast-cancer dataset from Weka, and performance is evaluated based on accuracy, precision, recall, training time, and confusion matrices.

## Usage
1. **Prepare the Dataset:**
   - The project uses the `breast-cancer.csv` dataset from Weka. Ensure the dataset is placed in the same directory as the Python files.

2. **Run the Baseline Model:**
   - Execute the script for the baseline Random Forest model:
     ```bash
     python no-optimization.py
     ```

3. **Run Bayesian Optimization on Random Forest:**
   - Execute the script that applies standard Bayesian Optimization:
     ```bash
     python bayesian_optimization.py
     ```

4. **Run Meta-Optimized Bayesian Optimization:**
   - Execute the script where Bayesian Optimization is recursively tuned:
     ```bash
     python optimized.py
     ```

5. **View Results:**
   - Check the console output for performance metrics (accuracy, precision, recall, runtime) and confusion matrices.
