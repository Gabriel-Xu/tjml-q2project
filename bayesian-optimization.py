# !pip install bayesian-optimization
# Bayesian optimization
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization
import time

def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=0,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred=model.predict(X_train)
    accuracy=accuracy_score(y_train, y_pred)
    return accuracy

param_bounds = {
    'n_estimators': (10, 200),
    'max_depth': (5, 50),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10)
}

start=time.time()
optimizer=BayesianOptimization(f=rf_cv, pbounds=param_bounds, random_state=0)
optimizer.maximize(init_points=10, n_iter=50)
end=time.time()
print("Time:", end-start)
best_params=optimizer.max['params']
best_params['n_estimators']=int(best_params['n_estimators'])
best_params['max_depth']=int(best_params['max_depth'])
best_params['min_samples_split']=int(best_params['min_samples_split'])
best_params['min_samples_leaf']=int(best_params['min_samples_leaf'])
model=RandomForestClassifier(**best_params, random_state=0, n_jobs=-1)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
precision=precision_score(y_test, y_pred, pos_label='recurrence-events')
recall=recall_score(y_test, y_pred, pos_label='recurrence-events')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
