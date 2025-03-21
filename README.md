# tjml-q2project

# Time-Optimized Bayesian Optimization of Random Forest Hyperparameters

**Authors:** Gabriel Xu and Andrew Chen  
**Date:** January 31, 2025  
**Project:** Machine Learning Quarter 2 Project  
**Advisor:** Dr. Yilmaz


---

```markdown
# Time-Optimized Bayesian Optimization of Random Forest Hyperparameters

## Project Overview
This project implements a meta-optimized approach to Bayesian Optimization for tuning hyperparameters of a Random Forest Classifier. Our method recursively tunes the hyperparameters of the Bayesian Optimization algorithm itself, resulting in improved classification accuracy and reduced hyperparameter search time compared to conventional methods. The approach is demonstrated on the breast-cancer dataset from Weka, and performance is evaluated based on accuracy, precision, recall, training time, and confusion matrices.


## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   *Ensure that your `requirements.txt` includes necessary packages such as `scikit-learn`, `pandas`, `numpy`, and any libraries used for Bayesian Optimization.*

## Usage
1. **Prepare the Dataset:**
   - The project uses the `breast-cancer.csv` dataset from Weka. Ensure the dataset is placed in the `data/` directory (or update the path in the code accordingly).
   - If necessary, convert categorical variables using one-hot encoding.

2. **Run the Baseline Model:**
   - Execute the script for the baseline Random Forest model:
     ```bash
     python baseline_random_forest.py
     ```

3. **Run Bayesian Optimization on Random Forest:**
   - Execute the script that applies standard Bayesian Optimization:
     ```bash
     python bayesian_optimization_rf.py
     ```

4. **Run Meta-Optimized Bayesian Optimization:**
   - Execute the script where Bayesian Optimization is recursively tuned:
     ```bash
     python meta_optimized_bayesian_optimization.py
     ```

5. **View Results:**
   - Check the console output for performance metrics (accuracy, precision, recall, runtime) and confusion matrices.
   - Additional visualizations or logs may be generated depending on your implementation.


```
