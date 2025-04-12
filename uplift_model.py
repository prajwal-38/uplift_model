import pandas as pd
import numpy as np
import lightgbm as lgb
from causalml.inference.meta import BaseXLearner, BaseTLearner, BaseSLearner
from causalml.metrics import qini_score
import joblib
import os

#Two-Model
def train_two_model_approach(X_train, y_train, t_train):
    """
    Args:
        X_train: Training features
        y_train: Target variable
        t_train: Treatment indicator
    """
    X_train_treatment = X_train[t_train == 1]
    y_train_treatment = y_train[t_train == 1]
    
    X_train_control = X_train[t_train == 0]
    y_train_control = y_train[t_train == 0]
    
    print("Training treatment model...")
    treatment_model = lgb.LGBMClassifier(
        n_estimators=200,  
        max_depth=5,      
        learning_rate=0.05,
        random_state=42,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        verbose=-1
    )
    treatment_model.fit(X_train_treatment, y_train_treatment)
    
    print("Training control model...")
    control_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        device='gpu',  #Use CPU if gpu not available
        gpu_platform_id=0,
        gpu_device_id=0,
        verbose=-1
    )
    control_model.fit(X_train_control, y_train_control)
    
    return {
        'treatment_model': treatment_model,
        'control_model': control_model
    }

#Meta-Learner

def train_metalearners(X_train, y_train, t_train):
    """
    (XLearner skipped due to long training time)
    
    Args:
        X_train: Training features
        y_train: Target variable
        t_train: Treatment indicator
    """
    lgb_params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'random_state': 42,
        'device': 'gpu',  #Using GPU
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbose': -1
    }
    
    #Create meta-learners
    metalearners = {
        'SLearner': BaseSLearner(
            learner=lgb.LGBMClassifier(**lgb_params)
        ),
        'TLearner': BaseTLearner(
            learner=lgb.LGBMClassifier(**lgb_params)
        )

    }
    
    for name, model in metalearners.items():
        print(f"Training {name} with GPU acceleration...")
        model.fit(X=X_train, treatment=t_train, y=y_train)
    
    return metalearners

#Evaluating models
def evaluate_models(two_model, metalearners, X_test, y_test, t_test):
    """
    Args:
        two_model: Dictionary with treatment and control models
        metalearners: Dictionary of meta-learners
        X_test: Test features
        y_test: Test target
        t_test: Test treatment indicator
    """
    treatment_model = two_model['treatment_model']
    control_model = two_model['control_model']
    
    #Two-Model predictions
    p_treated = treatment_model.predict_proba(X_test)[:, 1]
    p_control = control_model.predict_proba(X_test)[:, 1]
    uplift_two_model = p_treated - p_control
    
    #Meta-Learner predictions
    uplift_tl = metalearners['TLearner'].predict(X_test)
    uplift_sl = metalearners['SLearner'].predict(X_test)
    
    results = {}
    
    #calculate a simple uplift metric
    def calculate_uplift_metric(y, t, uplift_scores):
        #Convert to numpy arrays if they're pandas Series
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(t, 'values'):
            t = t.values
        if hasattr(uplift_scores, 'values'):
            uplift_scores = uplift_scores.values
            
        idx = np.argsort(-uplift_scores)
        sorted_y = y[idx]
        sorted_t = t[idx]
        
        treatment_rate = sorted_y[sorted_t == 1].mean() if np.any(sorted_t == 1) else 0
        control_rate = sorted_y[sorted_t == 0].mean() if np.any(sorted_t == 0) else 0
        
        return treatment_rate - control_rate
    
    results['Two-Model'] = calculate_uplift_metric(y_test, t_test, uplift_two_model)
    results['TLearner'] = calculate_uplift_metric(y_test, t_test, uplift_tl)
    results['SLearner'] = calculate_uplift_metric(y_test, t_test, uplift_sl)
    
    uplift_scores = {
        'Two-Model': uplift_two_model,
        'TLearner': uplift_tl,
        'SLearner': uplift_sl
    }
    
    return results, uplift_scores

#Comprehensive evaluation with multiple metrics
def comprehensive_evaluation(models, X_val, y_val, t_val):
    """Evaluate uplift models using multiple metrics"""
    metrics = {}
    
    for name, model in models.items():
        if name == 'Two-Model':
            treatment_model = model['treatment_model']
            control_model = model['control_model']
            p_treated = treatment_model.predict_proba(X_val)[:, 1]
            p_control = control_model.predict_proba(X_val)[:, 1]
            uplift_scores = p_treated - p_control
        else:
            uplift_scores = model.predict(X_val)
        
        #Calculate metrics
        metrics[name] = {
            'qini_score': qini_score(y_val, uplift_scores, t_val),
            'auuc': calculate_auuc(y_val, uplift_scores, t_val),
            'uplift_at_10': uplift_at_k_percent(y_val, uplift_scores, t_val, k=10),
            'treatment_rank_corr': treatment_rank_correlation(uplift_scores, t_val, y_val)
        }
    
    return metrics

def uplift_at_k_percent(y, uplift_scores, treatment, k=10):
    """Calculate uplift for top k% of population"""
    n = len(y)
    n_k = int(n * k / 100)
    indices = np.argsort(uplift_scores)[-n_k:]

    treatment_indices = indices[treatment[indices] == 1]
    control_indices = indices[treatment[indices] == 0]
    
    if len(treatment_indices) == 0 or len(control_indices) == 0:
        return 0
    
    treatment_conv_rate = y[treatment_indices].mean()
    control_conv_rate = y[control_indices].mean()
    
    return treatment_conv_rate - control_conv_rate

def calculate_auuc(y, uplift_scores, treatment):
    """Area Under the Uplift Curve"""
    indices = np.argsort(uplift_scores)[::-1]
    sorted_y = y[indices]
    sorted_treatment = treatment[indices]

    n = len(y)

    x_axis = np.linspace(0, 1, n)
    y_treatment = sorted_y[sorted_treatment == 1]
    y_control = sorted_y[sorted_treatment == 0]

    treatment_cum = np.cumsum(y_treatment) / max(1, len(y_treatment))
    control_cum = np.cumsum(y_control) / max(1, len(y_control))
    max_len = max(len(treatment_cum), len(control_cum))

    if len(treatment_cum) < max_len:
        treatment_cum = np.pad(treatment_cum, (0, max_len - len(treatment_cum)), 'edge')
    if len(control_cum) < max_len:
        control_cum = np.pad(control_cum, (0, max_len - len(control_cum)), 'edge')
    
    uplift_curve = treatment_cum - control_cum
    
    #Calculate AUC
    return np.trapz(uplift_curve, x_axis[:max_len])

def treatment_rank_correlation(uplift_scores, treatment, y):
    """Correlation between predicted uplift rank and actual treatment effect"""
    uplift_ranks = np.argsort(np.argsort(uplift_scores))
    

    treated_y = y[treatment == 1].mean()
    control_y = y[treatment == 0].mean()
    actual_effect = treated_y - control_y
    
    #Calculate correlation
    corr = np.corrcoef(uplift_ranks, y * treatment * actual_effect)[0, 1]
    return corr

#Save trained models
def save_models(two_model, metalearners, feature_cols, output_dir='models'):
    """Save trained models for later use"""
    os.makedirs(output_dir, exist_ok=True)
    
    #Save two-model approach
    treatment_model = two_model['treatment_model']
    control_model = two_model['control_model']
    joblib.dump(treatment_model, f'{output_dir}/treated_model.pkl')
    joblib.dump(control_model, f'{output_dir}/control_model.pkl')
    
    #Save meta-learners
    for name, model in metalearners.items():
        joblib.dump(model, f'{output_dir}/{name}.pkl')
    
    #Save feature columns
    joblib.dump(feature_cols, f'{output_dir}/feature_cols.pkl')
    
    print(f"Models saved to {output_dir}/")