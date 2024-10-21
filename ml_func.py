# can get the functions
from sklearn.linear_model import LogisticRegression
from sksurv.compare import compare_survival

# Ignore wanring
import warnings
warnings.filterwarnings("ignore")

# Basic process
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import math
import statistics as s
import scipy.stats as stats

# Parameters optimization
from sklearn.model_selection import GridSearchCV 

# Performance
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import logging

# Configure logging for better debugging
logging.basicConfig(
    filename='logfile.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def calculate_metrics(model, X, y):
    y_pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    metrics_dict = {
        "accuracy": (tp + tn) / (tn + fp + fn + tp),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "PPV": tp / (tp + fp),
        "NPV": tn / (tn + fn),
        "f1": metrics.f1_score(y, y_pred),
        "kappa": metrics.cohen_kappa_score(y, y_pred)
    }
    
    try:
        y_pred_proba = model.predict_proba(X)[:, 1]
        metrics_dict["auc"] = metrics.roc_auc_score(y, y_pred_proba)
    except AttributeError:
        metrics_dict["auc"] = "NA"
    
    return metrics_dict

def save_results(results, path, feature_name, suffix):
    filename = f"{path}/{feature_name}_{suffix}.csv"
    pd.DataFrame(results).to_csv(filename, index=False)

def perform_grid_search(clf, param_grid, X_train, y_train, num_folds):
    grid_clf = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=num_folds,
        refit="accuracy",
        scoring=calculate_metrics
    )
    return grid_clf.fit(X_train, y_train)

def parameter_tuning(value):
    # Unpack input values
    X_train, X_test, X_valid, y_train, y_test, y_valid = value[:6]
    num_folds, feature_name, y_stage, y_dfs, path = value[6:]

    # Set model parameters
    log_reg_params = {
        "class_weight": ['balanced', None],
        "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
        "fit_intercept": [True, False]
    }
    
    classifiers = {
        'LogisticRegression': (LogisticRegression(), log_reg_params)
    }

    results_train, results_test, results_valid = [], [], []

    for name, (clf, params) in classifiers.items():
        try:
            logging.info(f"Running GridSearchCV for {name}")
            grid_clf = perform_grid_search(clf, params, X_train, y_train, num_folds)
            
            # Evaluate model performance
            perform_train = calculate_metrics(grid_clf, X_train, y_train)
            perform_test = calculate_metrics(grid_clf, X_test, y_test)
            perform_valid = calculate_metrics(grid_clf, X_valid, y_valid)
            
            # Cross-validation scores and model coefficients
            cv_scores = cross_val_score(clf, X_train, y_train, cv=num_folds)
            coef = grid_clf.best_estimator_.coef_
            intercept = grid_clf.best_estimator_.intercept_
            
            # Update performance metrics with additional information
            for metrics in [perform_train, perform_test, perform_valid]:
                metrics.update({
                    "cv": np.mean(cv_scores),
                    "name": feature_name,
                    "model_name": name,
                    "best_params": grid_clf.best_params_,
                    "coef": coef,
                    "intercept": intercept
                })

            # Collect results
            results_train.append(perform_train)
            results_test.append(perform_test)
            results_valid.append(perform_valid)

        except Exception as e:
            logging.error(f"Error occurred during GridSearchCV: {e}")
            continue

    # Generate predictions for validation data
    X_pred = 1 / (1 + np.exp(-np.sum(X_valid * coef, axis=1) + intercept))

    # Analyze results with ANOVA
    data = pd.DataFrame({"gene": X_pred, "stage": y_stage})
    stage_means = {stage: data[data["stage"] == stage]["gene"].mean() for stage in range(1, 5)}
    f_stat, p_value_anova = stats.f_oneway(*(data[data["stage"] == i]["gene"] for i in range(1, 5)))

    # Chi-square test for survival comparison
    group_indicator = pd.DataFrame(X_pred) < np.median(X_pred)
    chi_sq, p_value_chi = compare_survival(y_dfs, group_indicator)  # type: ignore

    # Validate the order of means and significance tests
    if (list(stage_means.values()) == sorted(stage_means.values()) or
        list(stage_means.values()) == sorted(stage_means.values(), reverse=True)):
        
        if p_value_anova <= 0.05 and p_value_chi <= 0.05:
            save_results(results_train, path, feature_name, "train")
            save_results(results_test, path, feature_name, "test")
            save_results(results_valid, path, feature_name, "validation")
        else:
            logging.warning(f"{feature_name}: ANOVA or Chi-square test not satisfied")
    else:
        logging.warning(f"{feature_name}: Stage means are not in the correct order")