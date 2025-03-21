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

## Data
- **Dataset:** `breast-cancer.csv`
- **Source:** Weka dataset (originally from the UC Irvine Machine Learning Repository)
- **Preprocessing:** 
  - Categorical features (e.g., hormone receptor status) are one-hot encoded.
  - Discretization is applied to features like “menopause”, “tumor-size”, and “inv-nodes”.
  - The dataset is split into 80% training and 20% testing sets with class balance maintained.

## Experiments and Results
Three models are compared:
1. **Baseline Random Forest:** Manually chosen hyperparameters.
2. **Bayesian-Optimized Random Forest:** Uses Bayesian Optimization with predefined settings.
3. **Meta-Optimized Bayesian Optimization:** Recursively tunes Bayesian Optimization hyperparameters.

### Experimental Results (Example)
| Model                              | Runtime (sec) | Accuracy | Precision | Recall |
| ---------------------------------- | ------------- | -------- | --------- | ------ |
| Baseline Random Forest             | 0.24          | 0.7414   | 0.7692    | 0.4545 |
| Bayesian Optimization              | 33.59         | 0.7586   | 0.8333    | 0.4545 |
| Meta-Optimized Bayesian Optimization | 20.97         | 0.7586   | 0.8333    | 0.4545 |

Confusion matrices for each model are provided in the project paper.

## Contributions
- **Abstract:** Andrew
- **Introduction:** Gabriel
- **Related Work:** Andrew
- **Dataset and Features:** Gabriel
- **Methods:** Gabriel
- **Experiments/Results/Discussion:** Gabriel
- **Conclusion/Future Work:** Andrew
- **References/Bibliography:** Andrew

## References
1. Garrido-Merchán, E. C., & Jariego-Pérez, L. C. (2021). *Towards automatic Bayesian optimization: A first step involving acquisition functions*. [Lecture Notes in Computer Science](https://doi.org/10.1007/978-3-030-85713-4_16).
2. Lindauer, M., Feurer, M., Eggensperger, K., Biedenkapp, A., & Hutter, F. (2019). *Towards assessing the impact of Bayesian optimization’s own hyperparameters*. [IJCAI Workshop](https://doi.org/10.48550/arXiv.1908.06674).
3. Snoek, J., Larochelle, H., & Adams, R. P. (2012). *Practical Bayesian optimization of machine learning algorithms*. [NIPS Proceedings](https://proceedings.neurips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf).
4. Thornton, C., Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2013). *Auto-WEKA: Combined selection and hyperparameter optimization of classification algorithms*. [ACM SIGKDD](https://doi.org/10.1145/2487575.2487629).
5. Zhang, Z., Li, M., Wang, Y., & Chen, W. (2019). *Hyperparameter optimization for machine learning models based on Bayesian optimization*. [Journal of Computer Science and Technology](https://doi.org/10.1016/j.jcst.2019.03.002).
6. Xu, G. & Chen, A. (2025). *Time-Optimized Bayesian Optimization of Random Forest Hyperparamaters*. [Google Colab](https://colab.research.google.com/drive/1ClEoIvJTcNjGLrQscv-BAarumYDO5_pj?usp=sharing).
7. Frank, E., Hall, M. A., & Witten, I. H. (2016). *The WEKA Workbench*. In _Data Mining: Practical Machine Learning Tools and Techniques_ (4th ed.).
8. Zwitter, M. & Soklic, M. (1988). *Breast Cancer [Dataset]*. [UCI Machine Learning Repository](https://doi.org/10.24432/C51P4M).



```
